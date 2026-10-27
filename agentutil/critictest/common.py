# -*- coding: utf-8 -*-
"""critictest 公共层：数据口径、PER 结构 critic、训练目标变换、评价指标。

本模块只依赖项目里的**纯 Python / torch** 模块：
  * `model.Critic` —— PER 方法使用的 critic（`arch='v1'`，
    `x →16→100→ cat(a) →80→50→1`），本实验要求结构逐层一致；
  * `util.soft_update / hard_update` —— PER 的 target 网络更新口径；
  * `aifflor.aifflor_env.AifflorPhysics` —— 风场仿真（FLORIS）。

不启动 MATLAB，也不 import 任何 `*_main.py` / `*_myfunc.py`。

口径（全部可在 `make_dataset.py --help` / `README.md` 复查）
------------------------------------------------------------------
* 风速 `u`：环境守卫区间 `[3.5, 25]` m/s。
* 行为 `a`：5 维轴向因子，学习带 `[0, 0.5]`；物理层再把 `a < 0.025` 抬到
  `A_PHYS_LO = 0.025`（`aifflor_turbine` 的高斯尾流 `a=0` 奇点，见其模块注释）。
  数据集同时保存**原始 a**（网络输入）与 `a_eff`（真正进 FLORIS 的动作）。
* 风场出力 `P`：`AifflorPhysics.power()` 的口径 [MW]，aifflor 生产配置 =
  `calibrated=False` + `power_scale = LEVEL_COEF`（见 `aifflor_main.py` 的论证）。
* 训练目标 `s = 1000·P/u³`（"specific power"）。依据：aifflor 是准稳态 FLORIS，
  功率严格 ∝ u³（单机 `P = 0.95·½ρA·Cp·u_eff³`，尾流只改变 `u_eff/u` 的常数），
  实测同一动作下 `1000·P/u³` 在 u=3.5→25 只变化 0~9%（见 README 的表）。
  把精确的 u³ 因子提出去，网络只需拟合 5 维动作响应 `s(u,a)`，既可解释，
  也避免 16/100 的小网络把容量浪费在合成 u³ 上。
"""
import json
import os
import sys

# ---------------------------------------------------------------------------
# 路径 bootstrap（AGENTS.md 约定）：脚本可被任意 CWD / 任意路径调用
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))            # agentutil/critictest/
_AGENTUTIL = os.path.dirname(_HERE)                           # agentutil/
ROOT = os.path.dirname(_AGENTUTIL)                            # 仓库根
os.chdir(ROOT)
for _p in (ROOT, _AGENTUTIL, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from model import Critic  # noqa: E402
from util import hard_update, soft_update  # noqa: E402
from aifflor.aifflor_env import AifflorPhysics  # noqa: E402
from aifflor import aifflor_turbine as AT  # noqa: E402

# ---------------------------------------------------------------------------
# 口径常量
# ---------------------------------------------------------------------------
NT = 5                       # 机组数 = 动作维数
U_MIN, U_MAX = 3.5, 25.0     # env 的风速守卫
A_LO, A_HI = 0.0, 0.5        # 学习带（与 td3per/aifflor 的 CheckAction 相同）
A_PHYS_LO = float(AT.A_PHYS_LO)   # 0.025：aifflor 物理安全下界

S_SCALE = 1000.0             # s = S_SCALE · P / u³
P_SCALE = 10.0               # 直接拟合 P 时的目标缩放（消融臂用）
P_MAPE_FLOOR = 0.05          # MW：MAPE 分母下限（低风速 P→0 时防爆）

WIND_CSV = os.path.join(ROOT, 'FFC_speed_noise_1.csv')

TARGET_KINDS = ('sp', 'p', 'log')
FEATURE_KINDS = ('u10', 'u3')


# ---------------------------------------------------------------------------
# 随机性 / 线程
# ---------------------------------------------------------------------------
def set_seeds(seed):
    """固定 numpy / torch（CPU）随机源，保证数据与训练可复现。"""
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_threads(n):
    if n and int(n) > 0:
        torch.set_num_threads(int(n))


# ---------------------------------------------------------------------------
# 风场物理（aifflor）
# ---------------------------------------------------------------------------
def build_physics(calibrated=False, power_scale=None, verbose=False):
    """构造 aifflor 物理后端。

    Parameters
    ----------
    calibrated : bool
        `False`（默认）= aifflor **生产配置**：FLORIS 随包默认尾流参数 +
        `power_scale=LEVEL_COEF`（把电平抬到 WFSim，见 `aifflor_main.py`）。
        `True` = 用 `floris_calibrated_defaults` 的标定参数 + `power_scale=1`。
    power_scale : float | None
        None ⇒ 按 `calibrated` 自动选（False → `AT.LEVEL_COEF`，True → 1.0）。
    """
    if power_scale is None:
        power_scale = 1.0 if calibrated else float(AT.LEVEL_COEF)
    return AifflorPhysics(nt=NT, calibrated=bool(calibrated),
                          power_scale=float(power_scale), verbose=verbose)


def physics_meta(physics):
    """把 physics 的引擎口径压成可写入 JSON 的元数据。"""
    info = dict(getattr(physics, 'info', {}) or {})
    return dict(
        engine='aifflor/FLORIS',
        floris_version=info.get('floris_version'),
        calibrated=bool(physics.calibrated),
        wake_model=physics.wake_model,
        power_scale=float(physics.power_scale),
        model_strings=info.get('model_strings'),
        layout=info.get('layout'),
        rotor_diameter=info.get('rotor_diameter'),
        hub_height=info.get('hub_height'),
        rho=info.get('rho'),
        wind_shear=info.get('wind_shear'),
        ti=info.get('ti'),
        wind_direction=info.get('wind_direction'),
        u_min=U_MIN, u_max=U_MAX, a_lo=A_LO, a_hi=A_HI, a_phys_lo=A_PHYS_LO,
        nt=NT,
    )


def power_of(physics, u, a, chunk=4096, verbose=False):
    """批量算风场出力 [MW]；返回 `(P, a_eff)`。

    * 先按 `physics.a_clip` 把原始动作映射到真正进 FLORIS 的 `a_eff`
      （学习带 `[0,0.5]` → 物理下界 `0.025`）；
    * 分块调用 `AT.eval_batch`（FLORIS 批量调用一次别太大，省内存）；
    * `AT.eval_batch` 内部已做非有限值逐条重试 + 标称动作兜底；
      这里再断言一次有限，坏点必须显式暴露，不能静默进入数据集。
    """
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    a = np.asarray(a, dtype=np.float64).reshape(-1, NT)
    if u.size != a.shape[0]:
        raise ValueError('u 与 a 行数不一致: %d vs %d' % (u.size, a.shape[0]))

    a_eff = np.empty_like(a)
    for i, row in enumerate(a):
        a_eff[i] = physics.a_clip(row)

    stat0 = AT.stats()
    out = np.empty(u.size, dtype=np.float64)
    for s in range(0, u.size, int(chunk)):
        e = min(s + int(chunk), u.size)
        farm, _ = AT.eval_batch(physics.model, u[s:e], a_eff[s:e])
        out[s:e] = np.asarray(farm, dtype=np.float64) * float(physics.power_scale)
    stat1 = AT.stats()

    if not np.isfinite(out).all():
        n_bad = int((~np.isfinite(out)).sum())
        raise RuntimeError('aifflor 返回了 %d 个非有限值（本应被 eval_batch 兜底）' % n_bad)
    if verbose:
        print('[power] %d 点 | 物理层坏点 %d（重试救回 %d / 退回标称 %d）| 出口兜底 %d'
              % (u.size,
                 stat1['n_bad_in_batch'] - stat0['n_bad_in_batch'],
                 stat1['n_retry_ok'] - stat0['n_retry_ok'],
                 stat1['n_fallback'] - stat0['n_fallback'],
                 getattr(physics, 'n_nonfinite', 0)))
    return out, a_eff


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------
def sample_iid(n, rng):
    """i.i.d. 覆盖采样：u ~ U(3.5,25)，a ~ U(0,0.5)^5。"""
    n = int(n)
    u = rng.uniform(U_MIN, U_MAX, size=n)
    a = rng.uniform(A_LO, A_HI, size=(n, NT))
    return u.astype(np.float64), a.astype(np.float64)


def load_wind_pool(path=None):
    """读真实风速序列（`FFC_speed_noise_1.csv`，1440 行 × 200 天）。"""
    path = path or WIND_CSV
    df = pd.read_csv(path)
    arr = df.values.astype(np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    arr = arr[(arr >= U_MIN) & (arr <= U_MAX)]
    if arr.size == 0:
        raise RuntimeError('%s 里没有落在 [%.1f, %.1f] 的风速' % (path, U_MIN, U_MAX))
    return arr


def _family_actions(family, m, rng):
    """真实工况动作族（5 维），全部落在学习带内。"""
    idx = np.arange(NT, dtype=np.float64)
    if family == 'level':
        c = rng.uniform(0.05, 0.45, size=m)
        a = np.repeat(c[:, None], NT, axis=1)
    elif family == 'ramp':
        base = rng.uniform(0.05, 0.30, size=m)
        slope = rng.uniform(-0.040, 0.040, size=m)
        a = base[:, None] + slope[:, None] * idx[None, :]
    elif family == 'cp_opt':
        a = 1.0 / 3.0 + rng.normal(0.0, 0.04, size=(m, NT))
    elif family == 'farm_opt':
        a = 0.20 + rng.normal(0.0, 0.04, size=(m, NT))
    elif family == 'hetero':
        a = rng.uniform(0.03, 0.48, size=(m, NT))
    else:
        raise ValueError('未知动作族 %r' % (family,))
    return np.clip(a, 0.02, A_HI)


REAL_FAMILIES = ('level', 'ramp', 'cp_opt', 'farm_opt', 'hetero')


def sample_realistic(n_per_family, rng, wind_pool=None):
    """真实工况测试集：u 来自实测风速池，a 来自 5 种结构化动作族。

    与 i.i.d. 覆盖集的区别：u 的边缘分布是真实的（低风速占多数），动作是
    工程上有意义的形状（整场同水平 / 沿程斜坡 / 单机最优附近 / 场级最优附近 /
    逐台独立），因此能回答"在真实运行区域里拟合得怎样"。
    """
    pool = load_wind_pool() if wind_pool is None else np.asarray(wind_pool, float)
    us, acts, fams = [], [], []
    for j, fam in enumerate(REAL_FAMILIES):
        m = int(n_per_family)
        us.append(rng.choice(pool, size=m, replace=True))
        acts.append(_family_actions(fam, m, rng))
        fams.append(np.full(m, j, dtype=np.int16))
    u = np.concatenate(us)
    a = np.vstack(acts)
    fam = np.concatenate(fams)
    return u.astype(np.float64), a.astype(np.float64), fam


# ---------------------------------------------------------------------------
# 特征 / 目标变换
# ---------------------------------------------------------------------------
def state_features(u, kind='u10'):
    """把风速 u 变成 critic 的状态输入（确定性变换，不含任何未来信息）。

    * `u10`（默认）：`[u/10]`。与 PER 的状态归一化口径（`u/10`）一致。
    * `u3`：`[u/10, (u/10)², (u/10)³]`，显式给网络三次项（消融用；主臂不需要，
      因为 u³ 已在**目标**里被精确提出去）。
    """
    z = np.asarray(u, dtype=np.float64).reshape(-1) / 10.0
    if kind == 'u10':
        return z.reshape(-1, 1).astype(np.float32)
    if kind == 'u3':
        return np.column_stack([z, z * z, z * z * z]).astype(np.float32)
    raise ValueError('未知 state_feat %r' % (kind,))


def target_transform(p, u, kind='sp'):
    """训练目标（网络直接回归的量）。"""
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    if kind == 'sp':
        return (S_SCALE * p / np.power(u, 3)).astype(np.float32)
    if kind == 'p':
        return (p / P_SCALE).astype(np.float32)
    if kind == 'log':
        return np.log(p).astype(np.float32)
    raise ValueError('未知 target %r' % (kind,))


def target_to_power(out, u, kind='sp'):
    """把网络输出还原成风场出力 P [MW]（评价/部署都用这条）。"""
    out = np.asarray(out, dtype=np.float64).reshape(-1)
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    if kind == 'sp':
        return out * np.power(u, 3) / S_SCALE
    if kind == 'p':
        return out * P_SCALE
    if kind == 'log':
        return np.exp(out)
    raise ValueError('未知 target %r' % (kind,))


# ---------------------------------------------------------------------------
# 网络
# ---------------------------------------------------------------------------
def make_critic(ns, na=NT, hidden1=16, hidden2=100, init_w=3e-3, seed=None,
                bias=None):
    """PER 的 critic：`Critic(ns, na, hidden1=16, hidden2=100, arch='v1')`。

    `bias`：输出层 bias 初值。PER 生产路径用 `BiasInitTracker` 把 critic 输出层
    bias 设成价值基线；这里等价地把初值设成训练目标的均值（不是 0），以免前
    几千步都在"把输出搬到正确量级"。
    """
    if seed is not None:
        torch.manual_seed(int(seed))
    q = Critic(int(ns), int(na), hidden1=int(hidden1), hidden2=int(hidden2),
               init_w=float(init_w), arch='v1')
    if bias is not None:
        with torch.no_grad():
            q.fc3.bias.fill_(float(bias))
    return q


def make_critic_pair(ns, na=NT, hidden1=16, hidden2=100, init_w=3e-3,
                     seed=None, bias=None):
    """等价的 online critic + target critic（target 用 hard_update 同步初值）。"""
    q = make_critic(ns, na, hidden1, hidden2, init_w, seed, bias)
    qt = make_critic(ns, na, hidden1, hidden2, init_w, None, bias)
    hard_update(qt, q)
    return q, qt


def predict_power(model, u, a, target='sp', feature='u10', batch=8192):
    """用 critic 预测风场出力 [MW]（eval / no_grad）。"""
    model.eval()
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    a = np.asarray(a, dtype=np.float64).reshape(-1, NT)
    out = np.empty(u.size, dtype=np.float64)
    with torch.no_grad():
        for s in range(0, u.size, int(batch)):
            e = min(s + int(batch), u.size)
            x = torch.as_tensor(state_features(u[s:e], feature))
            aa = torch.as_tensor(a[s:e].astype(np.float32))
            out[s:e] = model([x, aa]).cpu().numpy().reshape(-1)
    return target_to_power(out, u, target)


def predict_target(model, u, a, feature='u10', batch=8192):
    """网络原始输出（目标空间；sp 臂下就是 specific power s）。"""
    model.eval()
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    a = np.asarray(a, dtype=np.float64).reshape(-1, NT)
    out = np.empty(u.size, dtype=np.float64)
    with torch.no_grad():
        for s in range(0, u.size, int(batch)):
            e = min(s + int(batch), u.size)
            x = torch.as_tensor(state_features(u[s:e], feature))
            aa = torch.as_tensor(a[s:e].astype(np.float32))
            out[s:e] = model([x, aa]).cpu().numpy().reshape(-1)
    return out


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------
def regression_metrics(pred, true, mape_floor=P_MAPE_FLOOR):
    """R² / RMSE / MAE / MAPE / 最大绝对误差 / 偏差。"""
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    if pred.size != true.size:
        raise ValueError('pred/true 长度不一致')
    err = pred - true
    sse = float(np.sum(err * err))
    sst = float(np.sum((true - true.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else float('nan')
    denom = np.maximum(np.abs(true), float(mape_floor))
    return dict(
        n=int(true.size),
        r2=r2,
        rmse=float(np.sqrt(np.mean(err * err))),
        mae=float(np.mean(np.abs(err))),
        mape=float(100.0 * np.mean(np.abs(err) / denom)),
        max_abs=float(np.max(np.abs(err))),
        bias=float(np.mean(err)),
        mean_true=float(true.mean()),
        mean_pred=float(pred.mean()),
    )


def per_bin_metrics(u, pred, true, edges):
    """按 u 分箱的指标（每箱一个 dict）。"""
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    rows = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        m = (u >= lo) & (u < hi)
        if not np.any(m):
            continue
        rec = regression_metrics(pred[m], true[m])
        rec.update(lo=lo, hi=hi)
        rows.append(rec)
    return rows


# ---------------------------------------------------------------------------
# 解析代理基线（不是网络，只用来给"网络拟合到了什么水平"提供参照）
# ---------------------------------------------------------------------------
def analytic_proxy_x(u, a_eff):
    """`X = u³·Σ_i 4a_i(1−a_i)²`（单机 Cp·u³ 的可加代理，忽略尾流）。"""
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    a = np.asarray(a_eff, dtype=np.float64).reshape(-1, NT)
    g = 4.0 * a * np.power(1.0 - a, 2)
    return np.power(u, 3) * g.sum(1)


def fit_analytic_k(u, a_eff, p):
    """最小二乘拟合 `P ≈ k·X` 的 k（只在 train split 上拟合）。"""
    X = analytic_proxy_x(u, a_eff)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    den = float(np.dot(X, X))
    return float(np.dot(X, p) / den) if den > 0 else 0.0


# ---------------------------------------------------------------------------
# 杂项
# ---------------------------------------------------------------------------
def save_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def fmt_metrics(m, name=''):
    return ('%-22s n=%6d  R²=%.6f  RMSE=%.4f  MAE=%.4f  MAPE=%.3f%%  max|e|=%.3f  bias=%+.4f'
            % (name, m['n'], m['r2'], m['rmse'], m['mae'], m['mape'],
               m['max_abs'], m['bias']))
