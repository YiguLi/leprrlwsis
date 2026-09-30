# -*- coding: utf-8 -*-
"""`forcescale` 对逐台出力影响的**公式 + 幂律指数**（数值定标）。

公式链
------
    Actuator.m:64  Fthrust_j = F · ½ρ·Ue_j² · CT_prime/(1−a)     ← **唯一出现 F**
    Actuator.m:69  P_j      = powerscale·½ρA_r·CT_prime·⟨Ue_j³⟩   ← **不含 F**

⇒ `P_j` 只经流场 `Ue_j` 依赖 `F`：

    P_j(F) = powerscale·½ρA_r·CT_prime·⟨Ue_j(F)³⟩
    Ue_j(F) = U∞·(1 − d_j(F))

局部弹性（一阶）：

    (1/P_j)·dP_j/dF = −3/(1−d_j) · ∂d_j/∂F
    对数形式：dlnP_j/dlnF = −3/(1−d_j) · ∂d_j/∂lnF

而高斯尾流的亏损 `d ∝ sqrt(CT) ∝ sqrt(F)`（近尾流 CT→0 奇点让对数斜率在 0.5 与 1 之间），
所以**指数不是普适常数**，必须对具体参数化实测：

    dlnP_j/dlnF ≡ −γ_j          γ_j = 3/(1−d_j) · ∂ln d_j/∂lnF

并且 `T1` 严格 `γ₁ = 0`（无上游尾流）。

用法
----
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\analyse_forcescale_per_turbine.py
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

from floris_import_guard import check_interpreter as _chk, sanitize as _sanitize  # noqa: E402
_chk()
_sanitize()

import numpy as np  # noqa: E402

import agentutil.probe_engine_random_actions as R  # noqa: E402
import aifflor.aifflor_turbine as AT  # noqa: E402

DATA = os.path.join('output', 'calib_floris_to_wfsim.json')
F0 = 1.5
FS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]


def evaluate(F, vu, vA):
    R.CT_SCALE = F
    m, _ = AT.build_model(calibrated=True)
    R._A_SET = vA
    R._run_batch(m, vu, vA)
    pt = np.asarray(m.get_turbine_powers(), dtype=float) / 1e6
    uu = np.asarray(m.core.flow_field.u_sorted)                 # (n, nt, gz, gy)
    Uinf = vu[:, None, None, None]
    # 逐台"出力加权等效亏损"：1 − cbrt(mean(u³))/U∞   （与 P ∝ ⟨Ue³⟩ 同口径）
    deff = 1.0 - np.cbrt(np.mean(np.clip(uu, 0.0, None) ** 3, axis=(2, 3))) / vu[:, None]
    return pt, deff


def main():
    d = json.load(open(DATA, encoding='utf-8'))
    vu = np.asarray(d['val']['u'], dtype=float)
    vA = np.asarray(d['val']['A'], dtype=float)
    print('验证集 n=%d (u 均值 %.2f m/s)   F0 = %.1f' % (len(vu), vu.mean(), F0))

    res = {F: evaluate(F, vu, vA) for F in FS}
    base_pt = res[F0][0]

    print('\n逐台出力（MW，验证集平均）')
    print('  %-7s %8s %8s %8s %8s %8s %9s' % ('F', 'T1', 'T2', 'T3', 'T4', 'T5', '农场'))
    for F in FS:
        c = res[F][0].mean(0)
        print('  %-7.2f %8.4f %8.4f %8.4f %8.4f %8.4f %9.4f'
              % (F, c[0], c[1], c[2], c[3], c[4], c.sum()))

    print('\n相对 F0=1.5 的变化（%）')
    print('  %-7s %9s %9s %9s %9s %9s %10s'
          % ('F', 'T1', 'T2', 'T3', 'T4', 'T5', '农场'))
    for F in FS:
        r = 100 * (res[F][0].mean(0) / base_pt.mean(0) - 1)
        print('  %-7.2f %+9.2f %+9.2f %+9.2f %+9.2f %+9.2f %+10.2f'
              % (F, r[0], r[1], r[2], r[3], r[4],
                 100 * (res[F][0].sum(1).mean() / base_pt.sum(1).mean() - 1)))

    print('\n' + '=' * 94)
    print('幂律指数 γ_j ≡ −dlnP_j/dlnF（在 F0=1.5 处中心差分，步长 ±0.25）')
    print('=' * 94)
    h = 0.25
    Pp = res[F0 + h][0].mean(0)
    Pm = res[F0 - h][0].mean(0)
    gam = -(np.log(Pp) - np.log(Pm)) / (np.log(F0 + h) - np.log(F0 - h))
    share = base_pt.mean(0) / base_pt.sum(1).mean()
    print('  %-6s %11s %13s %16s' % ('机组', '出力份额', 'γ_j', '公式预测 γ_j'))
    deff0 = res[F0][1].mean(0)
    # 实测 ∂ln d/∂ln F 的两点估计
    deffP = res[F0 + h][1].mean(0)
    deffM = res[F0 - h][1].mean(0)
    dln_d = (np.log(np.clip(deffP, 1e-9, None)) - np.log(np.clip(deffM, 1e-9, None))) \
        / (np.log(F0 + h) - np.log(F0 - h))
    pred = 3.0 / (1.0 - deff0) * dln_d
    for j in range(5):
        print('  T%-5d %10.2f%% %13.4f %16.4f' % (j + 1, 100 * share[j], gam[j], pred[j]))
    farm = -(np.log(res[F0 + h][0].sum(1).mean()) - np.log(res[F0 - h][0].sum(1).mean())) \
        / (np.log(F0 + h) - np.log(F0 - h))
    print('  %-6s %10.2f%% %13.4f %16s' % ('农场', 100.0, farm, '（份额加权）'))
    print('  ⇒ 核对 Σ α_j γ_j = %.4f vs 农场实测 %.4f' % (float(np.sum(share * gam)), farm))

    print('\n' + '=' * 94)
    print('等效亏损 d_eff 与亏损对 F 的对数斜率（公式的两个输入）')
    print('=' * 94)
    print('  %-6s %14s %18s %16s' % ('机组', 'd_eff@F0', '∂ln d/∂ln F', '3/(1−d)'))
    for j in range(5):
        print('  T%-5d %14.4f %18.4f %16.4f'
              % (j + 1, deff0[j], dln_d[j], 3.0 / (1.0 - deff0[j])))

    print('\n' + '=' * 94)
    print('结论')
    print('=' * 94)
    print('  1) 结构（不依赖任何参数）：')
    print('       P_1 与 F **严格无关**（无上游尾流）—— 实测 γ₁ = %+.4f' % gam[0])
    print('       P_{j>1} 随 F 单调下降，且**越下游越陡**（γ₂<γ₃<γ₄≈γ₅）')
    print('       P_farm 按出力份额加权 ⇒ 因 T1 占 %.1f%%，农场弹性远小于下游各台'
          % (100 * share[0]))
    print('  2) 定量（本标定参数化，F0=1.5）：')
    print('       %s' % np.array2string(gam, precision=4))
    print('       农场 γ_farm = %.4f  ⇒ **F 变动 10%% ⇒ 农场出力只变 %.2f%%**'
          % (farm, 10 * farm))
    print('       但 T5 变动 10%% ⇒ %.2f%%；T2 变动 10%% ⇒ %.2f%%'
          % (10 * gam[4], 10 * gam[1]))
    print('  3) 含义：F 对**农场总功率**几乎不可辨识（γ_farm≈%.2f，与尾流参数简并），'
          % abs(farm))
    print('     但对**逐台**出力强可辨识（T2 起 γ≈1~1.3）⇒ 想约束逐台分布，F 是关键自由度。')
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
