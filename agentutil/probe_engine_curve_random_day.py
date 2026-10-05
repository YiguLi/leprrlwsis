# -*- coding: utf-8 -*-
"""随机风速日 + 随机行为：**aifflor(FLORIS) 与 WFSim(MATLAB) 各出一条出力曲线**。

做的事（一句话）
--------------
从训练风速池（`FFC_speed_noise_1.csv` 的 195 个"日"列）里随机抽**一个整天**（1440 min），
在同一天上生成**同一串随机动作**（逐台轴向因子 `a ∈ [0,0.5]`，可设保持块长），
然后把**逐位相同的 (u, a) 序列**分别喂给两个引擎，各得到一条 `P(t)` 曲线。

两个引擎的口径（**唯一差异是引擎**）
----------------------------------
                     aifflor 侧                                 WFSim 侧
  入口               WindBatteryAIF.PowerOutput                  env.WindBattery.PowerOutput
  物理后端           AifflorPhysics(calibrated=False,            MATLAB `WFSim_port.m`
                     power_scale=LEVEL_COEF≈0.59895)            （Make_Ax_b.m 的 dt=Inf 稳态解）
  动作               a ∈ [0,0.5]（逐台），再抬到 A_PHYS_LO=0.025 a ∈ [0,0.5]（逐台）
  围栏               3.5 ≤ u ≤ 25 否则 0.0（**两边同一个 env.py 口径**）  同
  布局/密度/切变     单排 5 台 5D、D=100、ρ=1.20、shear=0.12       同（meshing.m）

⚠ 两点必须随曲线一起声明：
  1. **aifflor 侧有 `A_PHYS_LO=0.025` 的抬升**（`a∈[0,0.025)` 按 0.025 计，占学习带 5%）；
     本脚本会统计被抬升的"机组·分钟"数并写进 JSON。
  2. 绝对电平本就不同（`LEVEL_COEF` 就是为此标定的）⇒ 看的是**形状/比值**，不是谁大谁小。

输出
----
  <out>/tensorboard/        ★ 曲线用 TensorBoard 表示（默认；`--no-tb` 关闭）
      aifflor/   curve/P_farm(分钟级), curve/P_farm_nolift(诊断),
                 curve/blk/P_farm, curve/blk/P_farm_first（5 分钟级）
      wfsim/     同 aifflor 的 tag（分钟级 + 5 分钟级）
      wind/      curve/u
      actions/   action/a0..a4
      compare/   curve/ratio, curve/diff_mw, curve/abs_rel_err,
                 curve/blk/ratio_of_means, curve/blk/diff_mean, summary/*（两侧都跑时）
      ⇒ `aifflor/` 与 `wfsim/` 用**同一个 tag** 写进两个 run
        ⇒ TB 里是**一张图两条线**（分钟级与 5 分钟级各一张）
  <out>/curve.csv         ★ CSV 记录器（分钟级）：1440 行，1 分钟/行
  <out>/curve_5min.csv    ★ CSV 记录器（5 分钟级）：288 行，1 个上层周期/行，块内聚合
                             `*_mean`(5 分钟均值) / `*_first`(块首那一分钟 = 引擎按
                             5 分钟分辨率调一次的结果) / `*_last` / `*_min` / `*_max`
  <out>/curve.json        元信息（抽中的日列/两个种子/块长/物理口径/计数器）+ 统计量
  <out>/curve.png         可选（`--plot`）：静态图，仅两侧都跑时才生成

  两份 CSV 都是**增量落盘**（复用 `util.flush_csv_append_batch`，每 240 分钟行 / 每个
  5 分钟行立刻 flush）⇒ 长跑中途中断也已有数据；分钟级在**慢的那一趟（WFSim）**里边跑边写。

查看
----
    tensorboard --logdir="output/engine_curve_random_day/tensorboard"

用法（**需要 MATLAB 写工作区外的权限**；只在 FLORIS 侧自检可加 --no-matlab）
----
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\probe_engine_curve_random_day.py
    ... --day-seed 2026 --act-seed 7 --steps 1440 --block 1
    ... --block 5                 # 动作每 5 分钟才重抽（贴近 H 层决策周期）
    ... --no-matlab               # 只出 FLORIS 曲线（自检/无 MATLAB 时）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)
os.environ.setdefault('MPLCONFIGDIR', os.path.join(ROOT, 'output', '_mplcache'))

import numpy as np                                     # noqa: E402
import pandas as pd                                    # noqa: E402

import env as ENV                                      # noqa: E402
from aifflor.aifflor_env import AifflorPhysics, WindBatteryAIF   # noqa: E402
from aifflor import aifflor_turbine as AT              # noqa: E402
from util import flush_csv_append_batch                # noqa: E402  唯一增量落盘实现

WIND_CSV = 'FFC_speed_noise_1.csv'
NT = 5
MAX_E = 6.0
U_LO, U_HI = 3.5, 25.0
A_LO, A_HI = 0.0, 0.5


def pick_day(day_seed, steps):
    """按 test_five_days.sample_days 同一口径抽日（iloc 位置池 = 1..n_col-1）。"""
    df = pd.read_csv(WIND_CSV)
    pool = list(range(1, df.shape[1]))                 # 第 0 列是行号
    rng = np.random.RandomState(day_seed)
    col = int(rng.choice(pool, size=1, replace=False)[0])
    series = df.iloc[:steps, col].to_numpy(dtype=float)
    if series.size < steps:
        raise SystemExit('列 %d 只有 %d 行，不足 %d' % (col, series.size, steps))
    return col, str(df.columns[col]), series


def random_actions(act_seed, steps, block):
    """逐台 i.i.d. U[a_lo,a_hi]，可选每 block 分钟才重抽（保持 = 生产 H 层周期）。"""
    rng = np.random.RandomState(act_seed)
    raw = rng.uniform(A_LO, A_HI, size=(steps, NT))
    if block > 1:
        held = np.empty_like(raw)
        for i in range(steps):
            held[i] = raw[(i // block) * block]
        raw = held
    return raw


def run_floris(us, A, verbose=True, on_minute=None):
    """aifflor 生产物理：WindBatteryAIF + AifflorPhysics（含围栏/抬升/守卫/预取）。

    `on_minute(i, p_i, p_nolift_i)` 逐分钟回调（用于**增量落盘**）。
    不带 `A_PHYS_LO` 抬升的对照曲线**先整批算完**（一次 `eval_batch`，~1 s），
    这样回调时两条曲线都已就绪 ⇒ 分钟级 CSV 可以在本趟就写全。
    """
    phy = AifflorPhysics(nt=NT, calibrated=False, power_scale=AT.LEVEL_COEF)
    # 诊断曲线：关掉 A_PHYS_LO 抬升（学习带 [0,0.5] 内原样送引擎）。生产口径永远带抬升。
    _farm_nolift, _ = AT.eval_batch(phy.model, us, np.clip(A, A_LO, A_HI))
    p_nolift = np.asarray(_farm_nolift, dtype=float) * AT.LEVEL_COEF
    env_ai = WindBatteryAIF(MAX_E, NT, False, None, physics=phy)
    env_ai.wind_day = pd.Series(us)          # 让生产预取路径可用（数值与直算逐位相同）
    p = np.zeros(len(us), dtype=float)
    t0 = time.time()
    for i, u in enumerate(us):
        env_ai.count_wind = i
        env_ai.u = float(u)
        p[i] = float(env_ai.PowerOutput(float(u), A[i]))
        if on_minute is not None:
            on_minute(i, float(p[i]), float(p_nolift[i]))
        if verbose and (i + 1) % 200 == 0:
            print('  [aifflor] %4d/%d  (%.1f s)' % (i + 1, len(us), time.time() - t0),
                  flush=True)
    if verbose:
        print('  [aifflor] 完成 %.1f s | %s' % (time.time() - t0, env_ai.summary()), flush=True)
    return p, p_nolift, phy, env_ai


def run_wfsim(us, A, verbose=True, on_minute=None):
    """WFSim 生产物理：env.WindBattery.PowerOutput → MATLAB WFSim_port。

    `on_minute(i, p_i)` 逐分钟回调（用于**增量落盘**）。
    """
    import matlab
    import matlab.engine
    if verbose:
        print('  [WFSim] 启动 MATLAB 引擎 ...', flush=True)
    t0 = time.time()
    eng = matlab.engine.start_matlab()
    if verbose:
        print('  [WFSim] 引擎就绪 (%.1f s)' % (time.time() - t0), flush=True)
    eng.cd(ROOT, nargout=0)
    env_ws = ENV.WindBattery(MAX_E, NT, False, eng)
    p = np.zeros(len(us), dtype=float)
    t0 = time.time()
    for i, u in enumerate(us):
        p[i] = float(env_ws.PowerOutput(float(u), A[i]))
        if on_minute is not None:
            on_minute(i, float(p[i]))
        if verbose and (i + 1) % 200 == 0:
            print('  [WFSim] %4d/%d  (%.1f s)' % (i + 1, len(us), time.time() - t0),
                  flush=True)
    eng.quit()
    if verbose:
        print('  [WFSim] 完成 %.1f s' % (time.time() - t0), flush=True)
    return p


def load_tb_series(tb_root, run, tag):
    """从已有 TB logdir 读回一条 scalar 序列 → (steps, values)（按 step 去重，取最后写入）。

    一个 run 目录里可能有**多个 event 文件**（每次 SummaryWriter 都会新建一个，
    历史运行不会被覆盖）⇒ 必须按 step 去重，否则同一分钟会出现重复点。
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    d = os.path.join(tb_root, run)
    if not os.path.isdir(d):
        return None
    ea = EventAccumulator(d, size_guidance={'scalars': 0})   # 0 = 不限量
    ea.Reload()
    if tag not in ea.Tags()['scalars']:
        return None
    last = {}
    for e in ea.Scalars(tag):
        last[int(e.step)] = float(e.value)
    steps = np.array(sorted(last), dtype=int)
    return steps, np.array([last[int(s)] for s in steps], dtype=float)


def rebuild_csv_from_tb(out_dir, verbose=True):
    """**不跑引擎**，直接从已有 TB 日志重建两份 CSV（分钟级 + 5 分钟级）。

    用途：引擎侧已经跑过（比如 WFSim 那趟要 MATLAB、耗时 8 min），现在只是要升级
    CSV 口径（如补 5 分钟级）⇒ 从 TB 里把曲线读回来重算聚合即可，无需重跑引擎。
    同时把缺失的 5 分钟级 TB 标签（`curve/blk/*`）补写进 `wfsim` / `compare` run。
    """
    tb_root = os.path.join(out_dir, 'tensorboard')
    if not os.path.isdir(tb_root):
        raise SystemExit('找不到 TB 目录: {}'.format(tb_root))
    s_pf = load_tb_series(tb_root, 'aifflor', 'curve/P_farm')
    s_pn = load_tb_series(tb_root, 'aifflor', 'curve/P_farm_nolift')
    s_pw = load_tb_series(tb_root, 'wfsim', 'curve/P_farm')
    s_u = load_tb_series(tb_root, 'wind', 'curve/u')
    s_a = {j: load_tb_series(tb_root, 'actions', 'action/a%d' % j) for j in range(NT)}
    for nm, s in [('aifflor/curve/P_farm', s_pf), ('wind/curve/u', s_u)] + \
                 [('actions/action/a%d' % j, s_a[j]) for j in range(NT)]:
        if s is None:
            raise SystemExit('TB 里缺少 {} ⇒ 无法重建 CSV'.format(nm))
    steps = s_pf[0]
    for nm, s in [('wind/curve/u', s_u)] + [('actions/action/a%d' % j, s_a[j])
                                            for j in range(NT)] + \
                ([('aifflor/curve/P_farm_nolift', s_pn)] if s_pn is not None else []) + \
                ([('wfsim/curve/P_farm', s_pw)] if s_pw is not None else []):
        if not np.array_equal(s[0], steps):
            raise SystemExit('TB 各 run 的 step 轴不一致（{}）'.format(nm))
    n = len(steps)
    if verbose:
        print('=' * 96)
        print('  --from-tb：不跑引擎，从 TB 日志重建两份 CSV')
        print('=' * 96)
        print('  TB logdir : %s' % tb_root)
        print('  读回      : %d 分钟 × aifflor%s | 动作 a0..a4 | 风速'
              % (n, ' + wfsim' if s_pw is not None else '（无 wfsim run）'))

    rec = CsvRecorder(out_dir, block=5, flush_every=240, verbose=verbose)
    for k in range(n):
        row = {'minute': int(steps[k]), 'u': float(s_u[1][k]),
               **{'a%d' % j: float(s_a[j][1][k]) for j in range(NT)},
               'P_aifflor_mw': float(s_pf[1][k]),
               'P_aifflor_nolift_mw': float(s_pn[1][k]) if s_pn is not None else float('nan')}
        if s_pw is not None:
            w = float(s_pw[1][k])
            row['P_wfsim_mw'] = w
            row['diff_mw'] = float(s_pf[1][k]) - w
            row['ratio'] = (float(s_pf[1][k]) / w) if abs(w) > 1e-9 else float('nan')
        rec.push(row)
    n_min, n_blk = rec.close(verbose=verbose)
    write_block_tb(tb_root, rec.block_rows, has_wfsim=(s_pw is not None))
    if verbose:
        print('  已补写 TB 的 5 分钟级标签: run `aifflor`%s 的 `curve/blk/*`'
              % (' 与 `wfsim`/`compare`' if s_pw is not None else ''))
        if s_pw is not None:
            pa = np.array([r['P_aifflor_mean'] for r in rec.block_rows])
            pwm = np.array([r['P_wfsim_mean'] for r in rec.block_rows])
            rr = pa / pwm
            print('\n  5 分钟级汇总: 均值 aifflor %.4f MW vs WFSim %.4f MW | '
                  '块比值 min/中位/均值/max = %.4f/%.4f/%.4f/%.4f'
                  % (pa.mean(), pwm.mean(), rr.min(), np.median(rr), rr.mean(), rr.max()))
    return n_min, n_blk


def stats(pw, pf, us, inside):
    d = pf - pw
    big = inside & (pw > 0.5)
    ratio = np.divide(pf, pw, out=np.full_like(pf, np.nan), where=pw > 0.5)
    rel = np.divide(d, pw, out=np.full_like(d, np.nan), where=np.abs(pw) > 1e-9)
    s = dict(
        n=int(len(us)), n_inside=int(inside.sum()), n_outside=int((~inside).sum()),
        mean_wfsim_mw=float(pw.mean()), mean_aifflor_mw=float(pf.mean()),
        std_wfsim_mw=float(pw.std()), std_aifflor_mw=float(pf.std()),
        energy_wfsim_mwh=float(pw.sum() / 60.0), energy_aifflor_mwh=float(pf.sum() / 60.0),
        energy_ratio=float(pf.sum() / max(pw.sum(), 1e-12)),
        mean_diff_mw=float(d[inside].mean()) if inside.any() else None,
        mean_rel_diff=float(np.nanmean(rel[inside])) if inside.any() else None,
        median_rel_diff=float(np.nanmedian(rel[inside])) if inside.any() else None,
        mape=float(np.nanmean(np.abs(rel[inside]))) if inside.any() else None,
        median_ape=float(np.nanmedian(np.abs(rel[inside]))) if inside.any() else None,
        ratio_min=float(np.nanmin(ratio[big])) if big.any() else None,
        ratio_median=float(np.nanmedian(ratio[big])) if big.any() else None,
        ratio_mean=float(np.nanmean(ratio[big])) if big.any() else None,
        ratio_max=float(np.nanmax(ratio[big])) if big.any() else None,
        dp_wfsim_mean_mw=float(np.abs(np.diff(pw[inside])).mean()) if inside.sum() > 1 else None,
        dp_aifflor_mean_mw=float(np.abs(np.diff(pf[inside])).mean()) if inside.sum() > 1 else None,
    )
    if big.sum() > 5:
        s['corr'] = float(np.corrcoef(pw[big], pf[big])[0, 1])
        s['k_best'] = float(np.dot(pw[big], pf[big]) / np.dot(pw[big], pw[big]))
        resid = pf[big] - s['k_best'] * pw[big]
        s['resid_rms_mw'] = float(np.sqrt((resid ** 2).mean()))
    return s


class CsvRecorder:
    """**分钟级 + 5 分钟级两份 CSV** 的增量记录器（落盘复用 `util.flush_csv_append_batch`）。

    产物
    ----
    * `curve.csv`      —— 分钟级：1440 行，每行 1 分钟（u / a0..a4 / 两引擎出力 / Δ / 比值）
    * `curve_5min.csv` —— 5 分钟级：288 行（= 1440/5），每行 1 个**上层周期**，内容是该块
      5 分钟的**聚合**：`u_mean/first/last/min/max`、`a*_first`（块首决策值，= H 层实际下发的
      动作）、`a*_mean`、以及每个引擎的 `P_*_mean/first/last/min/max`、`diff_mean`、
      `ratio_of_means`（均值之比）、`ratio_mean`（逐分钟比值再平均）。

    关于"5 分钟级出力"的两种口径，这里把**两种都记进同一行**（避免歧义）：
      * `P_*_mean`  = 块内 5 个分钟的算术平均（"5 分钟平均出力"）
      * `P_*_first` = 块首那一分钟的引擎返回值。**在"引擎无记忆"的前提下**（WFSim 的
        `WFSim_port` 是 `dt=Inf` 稳态解、FLORIS 是准稳态闭式解，二者都无跨调用状态），
        它**逐位等于**"把引擎按 5 分钟分辨率调一次（动作保持整块）"的结果。
    两条口径给不同的东西，故都保留；末块不足 5 分钟时 `n_minutes` 会小于 5。
    """

    def __init__(self, out_dir, block=5, flush_every=240, verbose=True):
        self.dir = out_dir
        self.block = int(block)
        self.flush_every = int(flush_every)
        self.min_path = os.path.join(out_dir, 'curve.csv')
        self.blk_path = os.path.join(out_dir, 'curve_5min.csv')
        self._min_buf, self._min_written = [], 0
        self._blk_buf, self._blk_written = [], 0
        self._cur = []                      # 当前块内已收的分钟行
        self.block_rows = []                # 已落盘的 5 分钟行（供 TB 复用）
        for p in (self.min_path, self.blk_path):
            if os.path.exists(p):
                os.remove(p)                # 覆盖旧同名表（首次写表头）
        if verbose:
            print('  [CSV] 记录器就绪: %s (分钟级) + %s (5 分钟级, 每 %d 分钟一行)'
                  % (self.min_path, self.blk_path, self.block))

    # -- 分钟级 ---------------------------------------------------------------
    def push(self, rec):
        """喂一行**分钟级**记录（列必须与首行一致）；满 flush_every 行即落盘。"""
        self._min_buf.append(rec)
        self._cur.append(rec)
        if len(self._min_buf) >= self.flush_every:
            self._min_written = flush_csv_append_batch(
                self.min_path, self._min_buf, self._min_written, index=False)
            self._min_buf = []
        if len(self._cur) >= self.block:      # 一个上层周期收满 ⇒ 立刻出 5 分钟行
            row = self._rollup(self._cur)
            self._cur = []
            self._blk_buf.append(row)
            self.block_rows.append(row)
            self._blk_written = flush_csv_append_batch(
                self.blk_path, self._blk_buf, self._blk_written, index=False)
            self._blk_buf = []

    # -- 5 分钟级（块聚合） ----------------------------------------------------
    def _rollup(self, rows):
        b = len(self.block_rows)
        us = np.array([r['u'] for r in rows], dtype=float)
        Amat = np.array([[r['a%d' % j] for j in range(NT)] for r in rows], dtype=float)
        out = {'block': int(b), 'minute_start': int(rows[0]['minute']),
               'minute_end': int(rows[-1]['minute']), 'n_minutes': int(len(rows)),
               'u_mean': float(us.mean()), 'u_first': float(us[0]),
               'u_last': float(us[-1]), 'u_min': float(us.min()), 'u_max': float(us.max())}
        for j in range(NT):
            out['a%d_first' % j] = float(Amat[0, j])
            out['a%d_mean' % j] = float(Amat[:, j].mean())
        for key, name in (('P_aifflor_mw', 'P_aifflor'),
                          ('P_aifflor_nolift_mw', 'P_aifflor_nolift'),
                          ('P_wfsim_mw', 'P_wfsim')):
            if key in rows[0]:
                v = np.array([r[key] for r in rows], dtype=float)
                out[name + '_mean'] = float(v.mean())
                out[name + '_first'] = float(v[0])
                out[name + '_last'] = float(v[-1])
                out[name + '_min'] = float(v.min())
                out[name + '_max'] = float(v.max())
        if 'P_wfsim_mw' in rows[0]:
            pa = np.array([r['P_aifflor_mw'] for r in rows], dtype=float)
            pwm = np.array([r['P_wfsim_mw'] for r in rows], dtype=float)
            out['diff_mean'] = float((pa - pwm).mean())
            out['ratio_of_means'] = (float(pa.mean() / pwm.mean())
                                     if abs(pwm.mean()) > 1e-9 else float('nan'))
            rr = [a / w for a, w in zip(pa, pwm) if w > 1e-9]
            out['ratio_mean'] = float(np.mean(rr)) if rr else float('nan')
        return out

    # -- 收尾 ----------------------------------------------------------------
    def close(self, verbose=True):
        if self._min_buf:
            self._min_written = flush_csv_append_batch(
                self.min_path, self._min_buf, self._min_written, index=False)
            self._min_buf = []
        if self._cur:                         # 末块不足 block 分钟也记一行
            row = self._rollup(self._cur)
            self._cur = []
            self._blk_buf.append(row)
            self.block_rows.append(row)
        if self._blk_buf:
            self._blk_written = flush_csv_append_batch(
                self.blk_path, self._blk_buf, self._blk_written, index=False)
            self._blk_buf = []
        if verbose:
            print('  [CSV] 完成：%s = %d 行（分钟级）| %s = %d 行（5 分钟级）'
                  % (os.path.basename(self.min_path), self._min_written,
                     os.path.basename(self.blk_path), self._blk_written))
        return self._min_written, self._blk_written


def write_tensorboard(out_dir, us, A, pf, pf_nolift, pw, sumstats, meta, block_rows=None):
    """把曲线写进 TensorBoard。

    关键设计：两个引擎写成**两个 run 子目录**（`aifflor/`、`wfsim/`），但用
    **同一个 scalar tag `curve/P_farm`** ⇒ TB 的 scalar 面板里就是**一张图两条线**，
    可以直接叠加看两条出力曲线；风速、动作、比值各放自己的 run，互不干扰。
    """
    from torch.utils.tensorboard import SummaryWriter
    root = os.path.join(out_dir, 'tensorboard')
    os.makedirs(root, exist_ok=True)
    n = len(us)

    w = SummaryWriter(os.path.join(root, 'aifflor'))
    for i in range(n):
        w.add_scalar('curve/P_farm', float(pf[i]), i)
        w.add_scalar('curve/P_farm_nolift', float(pf_nolift[i]), i)
    for k, v in meta['lift_effect'].items():
        w.add_scalar('summary/lift_' + k, float(v), 0)
    w.add_scalar('summary/level_coef', float(meta['level_coef']), 0)
    w.add_scalar('summary/a_phys_lo', float(meta['a_phys_lo']), 0)
    w.flush()
    w.close()

    if pw is not None:                       # WFSim run（仅在跑了 MATLAB 时）
        w = SummaryWriter(os.path.join(root, 'wfsim'))
        for i in range(n):
            w.add_scalar('curve/P_farm', float(pw[i]), i)
        w.flush()
        w.close()

    w = SummaryWriter(os.path.join(root, 'wind'))       # 风速（两侧同一条）
    for i in range(n):
        w.add_scalar('curve/u', float(us[i]), i)
    w.flush()
    w.close()

    w = SummaryWriter(os.path.join(root, 'actions'))    # 随机动作（两侧同一串）
    for j in range(NT):
        for i in range(n):
            w.add_scalar('action/a%d' % j, float(A[i, j]), i)
    w.flush()
    w.close()

    if pw is not None:                       # 对照量（只有两侧都有才有定义）
        w = SummaryWriter(os.path.join(root, 'compare'))
        for i in range(n):
            w.add_scalar('curve/diff_mw', float(pf[i] - pw[i]), i)
            if pw[i] > 1e-9:
                w.add_scalar('curve/ratio', float(pf[i] / pw[i]), i)
            if abs(pw[i]) > 1e-9:
                w.add_scalar('curve/abs_rel_err',
                             abs(float(pf[i] - pw[i]) / pw[i]), i)
        for k in ('energy_ratio', 'mape', 'median_ape', 'mean_rel_diff',
                  'median_rel_diff', 'corr', 'k_best', 'resid_rms_mw',
                  'mean_wfsim_mw', 'mean_aifflor_mw', 'energy_wfsim_mwh',
                  'energy_aifflor_mwh'):
            if isinstance(sumstats, dict) and sumstats.get(k) is not None:
                w.add_scalar('summary/' + k, float(sumstats[k]), 0)
        w.flush()
        w.close()

    # ---- 5 分钟级曲线（x 轴 = 上层周期序号）------------------------------------
    if block_rows:
        write_block_tb(root, block_rows, has_wfsim=(pw is not None))
    return root


def write_block_tb(root, block_rows, has_wfsim):
    """把 **5 分钟级**曲线写进 TB（x 轴 = 上层周期序号 0..287）。

    与分钟级分开命名空间（`curve/blk/*`），避免 288 点与 1440 点混在一条 x 轴上；
    两个引擎仍用**同一个 tag** ⇒ 5 分钟分辨率下同样是"一张图两条线"。
    """
    from torch.utils.tensorboard import SummaryWriter
    w = SummaryWriter(os.path.join(root, 'aifflor'))
    for r in block_rows:
        w.add_scalar('curve/blk/P_farm', float(r['P_aifflor_mean']), int(r['block']))
        w.add_scalar('curve/blk/P_farm_first', float(r['P_aifflor_first']), int(r['block']))
        w.add_scalar('curve/blk/u', float(r['u_mean']), int(r['block']))
    w.flush()
    w.close()
    if not has_wfsim:
        return
    w = SummaryWriter(os.path.join(root, 'wfsim'))
    for r in block_rows:
        w.add_scalar('curve/blk/P_farm', float(r['P_wfsim_mean']), int(r['block']))
        w.add_scalar('curve/blk/P_farm_first', float(r['P_wfsim_first']), int(r['block']))
    w.flush()
    w.close()
    w = SummaryWriter(os.path.join(root, 'compare'))
    for r in block_rows:
        w.add_scalar('curve/blk/ratio_of_means', float(r['ratio_of_means']), int(r['block']))
        w.add_scalar('curve/blk/diff_mean', float(r['diff_mean']), int(r['block']))
    w.flush()
    w.close()


def make_plot(path, us, pw, pf, s, cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = np.arange(len(us)) / 60.0
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=False,
                                   gridspec_kw=dict(height_ratios=[2, 1]))
    ax1.plot(t, pw, lw=1.0, color='#1f77b4', label='WFSim (MATLAB)')
    ax1.plot(t, pf, lw=1.0, color='#d62728', label='aifflor (FLORIS x LEVEL_COEF)')
    ax1.set_ylabel('farm power [MW]')
    ax1.set_xlabel('hour of day')
    ax1.grid(alpha=0.3)
    axt = ax1.twinx()
    axt.plot(t, us, lw=0.8, ls=':', color='0.45', label='wind u [m/s]')
    axt.set_ylabel('u [m/s]', color='0.45')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = axt.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=9)
    ax1.set_title('random wind day (col {col}, seed {day_seed}) + random actions '
                  '(seed {act_seed}, block={block} min)\n'
                  'mean {mw:.3f} vs {mf:.3f} MW | energy ratio {er:.4f} | '
                  'corr {corr}'.format(col=cfg['wind_col'], day_seed=cfg['day_seed'],
                                       act_seed=cfg['act_seed'], block=cfg['block'],
                                       mw=s['mean_wfsim_mw'], mf=s['mean_aifflor_mw'],
                                       er=s['energy_ratio'],
                                       corr=('%.4f' % s['corr']) if 'corr' in s else 'n/a'),
                  fontsize=10)
    m = (us >= U_LO) & (us <= U_HI) & (pw > 0.5)
    r = pf[m] / pw[m]
    ax2.scatter(us[m], r, s=6, alpha=0.5, color='#2ca02c')
    ax2.axhline(1.0, color='0.4', lw=0.8, ls='--')
    if 'k_best' in s:
        ax2.axhline(s['k_best'], color='#d62728', lw=1.0,
                    label='k* = %.4f' % s['k_best'])
        ax2.legend(fontsize=9)
    ax2.set_xlabel('wind u [m/s]')
    ax2.set_ylabel('P_aifflor / P_wfsim')
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=125)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='随机日+随机行为：aifflor vs WFSim 各出一条出力曲线')
    ap.add_argument('--day-seed', type=int, default=2026, help='抽"哪一天"的随机种子')
    ap.add_argument('--act-seed', type=int, default=7, help='随机动作的种子')
    ap.add_argument('--steps', type=int, default=1440, help='分钟数（1440 = 完整一天）')
    ap.add_argument('--block', type=int, default=1, help='动作保持的分钟数（1=i.i.d.；5=H 层周期）')
    ap.add_argument('--out', default='output/engine_curve_random_day', help='输出目录')
    ap.add_argument('--no-matlab', action='store_true', help='只跑 aifflor(FLORIS) 侧')
    ap.add_argument('--no-tb', dest='tb', action='store_false',
                    help='不写 TensorBoard（默认写；曲线默认就用 TB 表示）')
    ap.add_argument('--plot', action='store_true',
                    help='额外输出静态 PNG（默认不写；两侧都跑了才有意义）')
    ap.add_argument('--from-tb', action='store_true',
                    help='不跑任何引擎：从已有 TB 日志重建 curve.csv / curve_5min.csv，'
                         '并把 5 分钟级标签补写进 TB（引擎侧已跑过时用这个升级 CSV 口径）')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.from_tb:                          # 纯离线：读 TB → 重算两份 CSV
        rebuild_csv_from_tb(args.out)
        print('\n  输出: %s  (curve.csv / curve_5min.csv 已重建)'
              % os.path.abspath(args.out))
        return 0
    t_start = time.time()
    print('=' * 96)
    print('  随机风速日 + 随机行为 → aifflor(FLORIS) / WFSim(MATLAB) 各一条出力曲线')
    print('=' * 96)

    col, colname, us = pick_day(args.day_seed, args.steps)
    A = random_actions(args.act_seed, args.steps, args.block)
    inside = (us >= U_LO) & (us <= U_HI)
    print('  抽中的日: FFC_speed_noise_1.csv 第 %d 列 (列名 %s)  n=%d  '
          'u∈[%.2f, %.2f] 均值 %.2f  (围栏 %.1f~%.1f 内 %d 条, 外 %d 条按 0 计)'
          % (col, colname, len(us), us.min(), us.max(), us.mean(), U_LO, U_HI,
             int(inside.sum()), int((~inside).sum())))
    print('  随机动作: seed=%d  block=%d min  a∈[%.2f,%.2f]^%d  '
          '实测范围 [%.4f, %.4f] 均值 %.4f'
          % (args.act_seed, args.block, A_LO, A_HI, NT, A.min(), A.max(), A.mean()))
    print('  物理口径: aifflor = AifflorPhysics(calibrated=False, power_scale=LEVEL_COEF=%.6f)'
          % AT.LEVEL_COEF)
    print('            （FLORIS %s；WFSim = MATLAB WFSim_port 稳态解）'
          % AT.build_model.__module__)

    # ---- CSV 记录器（分钟级 + 5 分钟级；增量落盘）----
    rec = CsvRecorder(args.out, block=5, flush_every=240)

    def _row(i, pf_i, pn_i, pw_i=None):
        r = {'minute': int(i), 'u': float(us[i]),
             **{'a%d' % j: float(A[i, j]) for j in range(NT)},
             'P_aifflor_mw': float(pf_i), 'P_aifflor_nolift_mw': float(pn_i)}
        if pw_i is not None:
            r['P_wfsim_mw'] = float(pw_i)
            r['diff_mw'] = float(pf_i) - float(pw_i)
            r['ratio'] = (float(pf_i) / float(pw_i)) if abs(float(pw_i)) > 1e-9 \
                else float('nan')
        return r

    t0 = time.time()
    pw = None
    if args.no_matlab:
        # 只有 aifflor 一趟 ⇒ 本趟就把分钟级写全（少两列：wfsim / Δ / 比值）
        def _cb_ai(i, pf_i, pn_i):
            rec.push(_row(i, pf_i, pn_i))
        pf, pf_nolift, phy, env_ai = run_floris(us, A, on_minute=_cb_ai)
    else:
        # 两趟：分钟级 CSV 在**第二趟（WFSim，慢）**里边跑边写 ⇒ 中途中断也已有数据
        pf, pf_nolift, phy, env_ai = run_floris(us, A)

        def _cb_ws(i, pw_i):
            rec.push(_row(i, pf[i], pf_nolift[i], pw_i))
        pw = run_wfsim(us, A, on_minute=_cb_ws)
    print('  [aifflor] 曲线完成（%.1f s）' % (time.time() - t0))
    print('  [aifflor] 诊断：不带 A_PHYS_LO 抬升的对照曲线（平均抬升 %+.4f MW，'
          '%d/%d 分钟受影响）'
          % ((pf - pf_nolift).mean(), int((np.abs(pf - pf_nolift) > 1e-9).sum()), len(us)))
    if args.no_matlab:
        print('  [WFSim] 已按 --no-matlab 跳过')
    n_min_rows, n_blk_rows = rec.close()

    # ---- 统计 ----
    sumstats = {}
    if pw is not None:
        sumstats = stats(pw, pf, us, inside)

    # ---- 落盘：两份 CSV 已由记录器增量写完（见 CsvRecorder）；这里只写 JSON ----
    lifted = int((A < AT.A_PHYS_LO).sum())
    meta = dict(
        wind_csv=WIND_CSV, wind_col=col, wind_col_name=colname,
        day_seed=args.day_seed, act_seed=args.act_seed, steps=args.steps,
        block_min=args.block, a_lo=A_LO, a_hi=A_HI, nt=NT,
        a_phys_lo=float(AT.A_PHYS_LO),
        n_turbine_minute_lifted=int(lifted),
        frac_turbine_minute_lifted=float(lifted / max(A.size, 1)),
        level_coef=float(AT.LEVEL_COEF),
        fence=[U_LO, U_HI],
        csv_minute=dict(path=os.path.join(args.out, 'curve.csv'), rows=int(n_min_rows),
                        granularity_min=1),
        csv_5min=dict(path=os.path.join(args.out, 'curve_5min.csv'), rows=int(n_blk_rows),
                      granularity_min=5,
                      note='块内聚合：*_mean=5 分钟均值；*_first=块首那一分钟的值'
                           '（引擎无记忆时 = 5 分钟分辨率单次调用的结果）'),
        aifflor_side='WindBatteryAIF + AifflorPhysics(calibrated=False, power_scale=LEVEL_COEF)',
        wfsim_side='env.WindBattery -> MATLAB WFSim_port',
        aifflor_physics_summary=env_ai.summary(),
        aifflor_counters=dict(n_batch=phy.n_batch, n_direct=phy.n_direct,
                              n_hit=phy.n_hit, n_nonfinite=phy.n_nonfinite,
                              n_cache_bad=phy.n_cache_bad,
                              prefetch=env_ai.n_prefetch),
        summary=sumstats,
        lift_effect=dict(
            mean_gain_mw=float((pf - pf_nolift).mean()),
            max_gain_mw=float((pf - pf_nolift).max()),
            mean_gain_pct=float(100.0 * (pf - pf_nolift).mean() / max(pf.mean(), 1e-12)),
            n_minutes_changed=int((np.abs(pf - pf_nolift) > 1e-9).sum())),
        elapsed_s=round(time.time() - t_start, 1),
    )
    with open(os.path.join(args.out, 'curve.json'), 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    tb_root = None
    if args.tb:                                  # ★ 曲线默认用 TensorBoard 表示
        tb_root = write_tensorboard(args.out, us, A, pf, pf_nolift, pw, sumstats, meta,
                                    block_rows=rec.block_rows)
    if args.plot and pw is not None:             # PNG 仅在显式要求且两侧都跑时输出
        make_plot(os.path.join(args.out, 'curve.png'), us, pw, pf, sumstats, meta)

    # ---- 打印 ----
    print('\n' + '=' * 96)
    print('  汇总' + ('' if pw is not None else '（仅 aifflor 侧）'))
    print('=' * 96)
    if pw is not None:
        print('  %-28s %14s %14s' % ('', 'WFSim', 'aifflor'))
        print('  %-28s %14.4f %14.4f' % ('全天电量 [MWh] (ΣP/60)',
                                         sumstats['energy_wfsim_mwh'],
                                         sumstats['energy_aifflor_mwh']))
        print('  %-28s %14.6f %14.6f' % ('平均出力 [MW]', sumstats['mean_wfsim_mw'],
                                         sumstats['mean_aifflor_mw']))
        print('  %-28s %14.6f %14.6f' % ('出力标准差 [MW]', sumstats['std_wfsim_mw'],
                                         sumstats['std_aifflor_mw']))
        print('  %-28s %14.6f %14.6f' % ('逐分钟 |ΔP| 均值 [MW]',
                                         sumstats['dp_wfsim_mean_mw'] or float('nan'),
                                         sumstats['dp_aifflor_mean_mw'] or float('nan')))
        print('  %-28s %14.4f' % ('电量比 (aifflor/WFSim)', sumstats['energy_ratio']))
        print('  %-28s %14.4f' % ('平均相对差 mean(Δ/P)', sumstats['mean_rel_diff']))
        print('  %-28s %14.4f' % ('中位相对差 median(Δ/P)', sumstats['median_rel_diff']))
        print('  %-28s %14.4f' % ('MAPE (围栏内)', sumstats['mape']))
        print('  %-28s %14.4f' % ('MedAPE (围栏内)', sumstats['median_ape']))
        if 'corr' in sumstats:
            print('  %-28s %14.6f' % ('corr(P_W, P_F)', sumstats['corr']))
            print('  %-28s %14.6f' % ('k* = argmin||P_F−k·P_W||', sumstats['k_best']))
            print('  %-28s %14.4f MW' % ('残差 RMS（抬平后）', sumstats['resid_rms_mw']))
        print('  %-28s %s' % ('比值 min/中位/均值/max',
                              '/'.join('%.4f' % sumstats[k] for k in
                                       ('ratio_min', 'ratio_median', 'ratio_mean',
                                        'ratio_max'))))
    print('  %-28s %d 个机组·分钟 (%.3f%%) 被抬到 A_PHYS_LO=%.3f'
          % ('aifflor 侧动作抬升', lifted, 100 * lifted / max(A.size, 1), AT.A_PHYS_LO))
    print('  %-28s 影响 %d/%d 分钟  平均 %+.4f MW (%+.3f%%)  最大 %+.4f MW'
          % ('  └ 抬升对曲线的作用', meta['lift_effect']['n_minutes_changed'], len(us),
             meta['lift_effect']['mean_gain_mw'], meta['lift_effect']['mean_gain_pct'],
             meta['lift_effect']['max_gain_mw']))
    print('  %-28s %s' % ('aifflor 计数器', meta['aifflor_counters']))
    print('  %-28s %d 行（分钟级，1 分钟/行）' % ('curve.csv', n_min_rows))
    print('  %-28s %d 行（5 分钟级，1 上层周期/行；块内聚合 *_mean + 块首 *_first）'
          % ('curve_5min.csv', n_blk_rows))
    if tb_root is not None:
        print('\n  ★ 曲线（TensorBoard）: %s' % tb_root)
        print('      分钟级：run `aifflor` 与 run `wfsim` 的 tag `curve/P_farm`（一张图两条线）'
              + ('' if pw is not None else '（run `wfsim` 待 MATLAB 那次运行写入）'))
        print('      5 分钟级：同名 run 的 tag `curve/blk/P_farm`（x 轴 = 上层周期序号）')
        print('      查看: tensorboard --logdir="%s"' % tb_root)
    print('  输出: %s  (curve.csv / curve_5min.csv / curve.json%s)'
          % (args.out, ' / curve.png' if (args.plot and pw is not None) else ''))
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
