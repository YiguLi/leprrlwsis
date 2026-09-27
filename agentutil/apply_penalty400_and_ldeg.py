# -*- coding: utf-8 -*-
"""两处改动 (用户 2026-09 指定):

① `penalty_NN_H` 的 200 元/单位 → **400 元/单位** (六个 TD3 方法的 H 层动作越界罚)。
   实现: 在各 `*_myfunc.py` 的 `def CheckAction` 前加两个具名常量并按分支替换字面量 ——
   `PENALTY_NN_H_COEF = 400.0` (H 层: 机组限功率 [0,0.5] ×5 + 信号 [±0.5])、
   `PENALTY_NN_L_COEF = 200.0` (下层 1 维, 仅诊断, **不变**)。
   判据: 该行含 `charge` ⇒ 下层分支; 否则 ⇒ H 层分支。
   换算: s_p=1000 ⇒ critic 空间 0.4/单位 (经济斜率仍 0.27~4.5) —— 只翻倍, 不改变量级结论。

② 为**下层 BESS 智能体**加"放电退化成本罚": `c2 = L_TRACK_COEF·trackerr + L_DEG_COEF·P_D`
   (P_D = max(0, −pbrule), MW; 一步 = 1 min ⇒ 两侧同为 **MW·min** 口径)。
   **设计约束 (用户)**: 放电退化罚必须比 SOC 跟踪信号**小 5~10 倍 (每 MW·min)**
   ⇒ `L_DEG_COEF ∈ [0.1, 0.2]`, 取 **0.15 (= 1/6.67)**; 与跟踪项同处 c2 通道
   ⇒ 比值 `L_DEG_COEF/L_TRACK_COEF` 与 λ2 无关 (λ2 同时乘两项), 5~10 倍在任何 λ2 下都成立。
   只在**真有 RL 下层 agent 的三个项目**接线 (td3per / td3prrbb / td3prle);
   另三个 (td3perrb / td3prrlrb / td3leprrlrb) 是规则下层, 无 L agent 可罚 ⇒ 不动。
   默认 `pbrule=None` ⇒ 不加该项 ⇒ 其余调用点与历史逐位不变。

用法: python agentutil/apply_penalty400_and_ldeg.py [--apply]
"""

import argparse
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except AttributeError:
    pass

SIX = ['td3per', 'td3perrb', 'td3prrlrb', 'td3prrbb', 'td3prle', 'td3leprrlrb']
RL_LOWER = ['td3per', 'td3prrbb', 'td3prle', 'prle']      # 真有 agent_L 且共用 hst.lower_cost_soc_track

CONST_BLOCK = '''PENALTY_NN_H_COEF = 400.0  # [元/单位] H 层动作越界罚 (机组限功率 [0,0.5]×5 + 信号 [±0.5])
                         #   原 200 -> 400: 越界仍未压住, 先整体加倍。
                         #   ⚠ s_p=1000 ⇒ critic 空间 0.4/单位, 而框内经济斜率实测 0.27~4.5
                         #   ⇒ 仍然同量级, 不改变"边界位置落在 critic 噪声里"的结论。
PENALTY_NN_L_COEF = 200.0  # [元/单位] 下层 1 维动作罚 —— **保持 200 不变** (仅诊断, 不进 r2 公式)


'''

CHECK_ANCHOR = 'def CheckAction('

# ---- ② 的接线锚点 (逐项目) ----
C2_SITES = {
    'td3per': [
        ('td3per/td3per_main.py',
         "            rbb_info_obs['pbrulemin'], rbb_info_obs['pbrulemax'],\n"
         "            5 - (episode_steps % 5),\n"
         "            return_info=True)\n",
         "            rbb_info_obs['pbrulemin'], rbb_info_obs['pbrulemax'],\n"
         "            5 - (episode_steps % 5),\n"
         "            pbrule=float(rbb_info['pbrule']),   # 放电退化罚 (L_DEG_COEF)\n"
         "            return_info=True)\n"),
        ('td3per/td3per_myfunc.py',
         "            rbb_info_obs['pbrulemin'], rbb_info_obs['pbrulemax'],\n"
         "            5 - (step % 5),\n"
         "            return_info=True)\n",
         "            rbb_info_obs['pbrulemin'], rbb_info_obs['pbrulemax'],\n"
         "            5 - (step % 5),\n"
         "            pbrule=float(rbb_info['pbrule']),   # 放电退化罚 (L_DEG_COEF)\n"
         "            return_info=True)\n"),
    ],
    'td3prrbb': [
        ('td3prrbb/td3prrbb_myfunc.py',
         "        rbb_info['pbrulemin'], rbb_info['pbrulemax'], n_rem, return_info=True)\n",
         "        rbb_info['pbrulemin'], rbb_info['pbrulemax'], n_rem,\n"
         "        pbrule=float(rbb_info['pbrule']),   # 放电退化罚 (L_DEG_COEF)\n"
         "        return_info=True)\n"),
    ],
    'td3prle': [
        ('td3prle/td3prle_myfunc.py',
         "        rbb_info['pbrulemin'], rbb_info['pbrulemax'], n_rem, return_info=True)\n",
         "        rbb_info['pbrulemin'], rbb_info['pbrulemax'], n_rem,\n"
         "        pbrule=float(rbb_info['pbrule']),   # 放电退化罚 (L_DEG_COEF)\n"
         "        return_info=True)\n"),
    ],
    # prle (非 TD3, 但同样有 RL 下层 agent 且共用同一个 c2 函数)
    'prle': [
        ('prle/prle_myfunc.py',
         "        rbb_info['pbrulemin'], rbb_info['pbrulemax'], n_rem, return_info=True)\n",
         "        rbb_info['pbrulemin'], rbb_info['pbrulemax'], n_rem,\n"
         "        pbrule=float(rbb_info['pbrule']),   # 放电退化罚 (L_DEG_COEF)\n"
         "        return_info=True)\n"),
    ],
}

SOCT_CONST_OLD = ('L_TRACK_COEF = 1.0       # c2 = +L_TRACK_COEF · trackerr           [MW]\n'
                  'L_SOC_COEF = L_TRACK_COEF   # 兼容旧名（旧口径里它是 MWh→代价的系数）\n')
SOCT_CONST_NEW = ('L_TRACK_COEF = 1.0       # c2 = +L_TRACK_COEF · trackerr           [MW]\n'
                  'L_SOC_COEF = L_TRACK_COEF   # 兼容旧名（旧口径里它是 MWh→代价的系数）\n'
                  'L_DEG_COEF = 0.15        # 放电退化成本系数: c2 += L_DEG_COEF · max(0, −pbrule)\n'
                  '#   [MW·min 口径] P_D 单位 MW、一步 = 1 min ⇒ 与 trackerr 同口径, 可直接比。\n'
                  '#   **设计约束**: 必须比 SOC 跟踪信号 (L_TRACK_COEF = 1.0) 小 5~10 倍 (每 MW·min)\n'
                  '#   ⇒ L_DEG_COEF ∈ [0.1, 0.2], 取 0.15 (= 1/6.67)。同处 c2 通道 ⇒ 比值与 λ2 无关\n'
                  '#   (λ2 同时乘两项), 5~10 倍在 λ2 ∈ [0.5, 2] 全域都成立。\n')

SOCT_SIG_OLD = ('def lower_cost_soc_track(soc_target, SOC_prev, SOC_next, pbrulemin, pbrulemax, n_rem,\n'
                '                         coef=None, return_info=False, **kw):\n')
SOCT_SIG_NEW = ('def lower_cost_soc_track(soc_target, SOC_prev, SOC_next, pbrulemin, pbrulemax, n_rem,\n'
                '                         coef=None, return_info=False,\n'
                '                         pbrule=None, deg_coef=None, **kw):\n')

SOCT_TAIL_OLD = ('    c = L_TRACK_COEF if coef is None else float(coef)\n'
                 '    cost = c * trackerr\n'
                 '    if return_info:\n'
                 '        return cost, info\n'
                 '    return cost\n')
SOCT_TAIL_NEW = ('    c = L_TRACK_COEF if coef is None else float(coef)\n'
                 '    cost = c * trackerr\n'
                 '    deg = 0.0\n'
                 '    if pbrule is not None:\n'
                 '        # 放电退化: 只对放电 (pbrule<0) 计费, 单位 MW·min (一步 = 1 min)\n'
                 '        cd = L_DEG_COEF if deg_coef is None else float(deg_coef)\n'
                 '        deg = cd * max(0.0, -float(pbrule))\n'
                 '        cost += deg\n'
                 '    if return_info:\n'
                 '        info = dict(info)\n'
                 '        info[\'c2_track\'] = c * trackerr\n'
                 '        info[\'c2_deg\'] = deg\n'
                 '        info[\'deg_coef\'] = (L_DEG_COEF if deg_coef is None else float(deg_coef))\n'
                 '        return cost, info\n'
                 '    return cost\n')

SOCT_DOC_OLD = ('    """下层 c2 = +coef · trackerr（两区间分治；单位 MW，与 r2 一致）。\n'
                '\n'
                '    `coef=None` ⇒ 取 `L_TRACK_COEF`（=1.0）。\n')
SOCT_DOC_NEW = ('    """下层 c2 = +coef · trackerr (+ L_DEG_COEF·P_D)（两区间分治；单位 MW，与 r2 一致）。\n'
                '\n'
                '    `coef=None` ⇒ 取 `L_TRACK_COEF`（=1.0）。\n'
                '    `pbrule=None` ⇒ **不加放电退化项**（历史口径逐位不变）；给了则加\n'
                '    `L_DEG_COEF·max(0, −pbrule)` —— 设计上比跟踪项小 5~10 倍（每 MW·min）。\n')


def read(p):
    with open(p, encoding='utf-8') as f:
        return f.read()


def write(p, s):
    with open(p, 'w', encoding='utf-8', newline='') as f:
        f.write(s)


def rep_once(path, old, new, label, dry, src_holder=None):
    src = src_holder[0] if src_holder else read(path)
    n = src.count(old)
    if n != 1:
        print('    [FAIL] %s: %s 命中 %d 次 (要求 1)' % (os.path.basename(path), label, n))
        return None
    src = src.replace(old, new)
    if src_holder:
        src_holder[0] = src
    else:
        if not dry:
            write(path, src)
    print('    [OK]   %s: %s' % (os.path.basename(path), label))
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    dry = not args.apply
    ok = True

    # ---------------- ① penalty_NN_H 200 -> 400 ----------------
    print('=== ① penalty_NN_H 200 -> 400 (H 层分支) ===')
    for k in SIX:
        p = '%s/%s_myfunc.py' % (k, k)
        src = read(p)
        if 'PENALTY_NN_H_COEF' in src:
            print('    [SKIP] %s: 已应用' % os.path.basename(p))
            continue
        n_h = n_l = 0
        out = []
        for line in src.split('\n'):
            if 'penalty +=' in line and ' 200' in line:
                if 'charge' in line:
                    line = line.replace('200', 'PENALTY_NN_L_COEF'); n_l += 1
                elif ('0.5' in line) or ('SIGNAL' in line) or ('sig' in line):
                    line = line.replace('200', 'PENALTY_NN_H_COEF'); n_h += 1
            out.append(line)
        src = '\n'.join(out)
        if n_h < 3 or n_l < 1:
            print('    [FAIL] %s: H 分支替换 %d 处 / L 分支 %d 处 (期望 ≥3 / ≥1)'
                  % (os.path.basename(p), n_h, n_l))
            ok = False
            continue
        if src.count(CHECK_ANCHOR) != 1:
            print('    [FAIL] %s: def CheckAction 出现 %d 次' % (os.path.basename(p), src.count(CHECK_ANCHOR)))
            ok = False
            continue
        src = src.replace(CHECK_ANCHOR, CONST_BLOCK + CHECK_ANCHOR)
        if not dry:
            write(p, src)
        print('    [OK]   %s: 常量已加, H 分支 %d 处 → PENALTY_NN_H_COEF(400), L 分支 %d 处 → PENALTY_NN_L_COEF(200)'
              % (os.path.basename(p), n_h, n_l))

    # ---------------- ② 下层放电退化罚 ----------------
    print('\n=== ② hsrl_soctarget: L_DEG_COEF + lower_cost_soc_track ===')
    p = 'hsrl_soctarget.py'
    src = read(p)
    if 'L_DEG_COEF' in src:
        print('    [SKIP] 已应用')
    else:
        for old, new, lab in ((SOCT_CONST_OLD, SOCT_CONST_NEW, '常量 L_DEG_COEF=0.15'),
                              (SOCT_SIG_OLD, SOCT_SIG_NEW, '签名 + pbrule/deg_coef'),
                              (SOCT_DOC_OLD, SOCT_DOC_NEW, 'docstring'),
                              (SOCT_TAIL_OLD, SOCT_TAIL_NEW, 'body: 加退化项 + info 分解')):
            if rep_once(p, old, new, lab, dry) is None:
                ok = False
    print('    --- 接线 (只给真有 RL 下层的三个项目) ---')
    for k in RL_LOWER:
        for path, old, new in C2_SITES[k]:
            if 'pbrule=float(rbb_info' in read(path):
                print('    [SKIP] %s: 已接线' % os.path.basename(path))
                continue
            if rep_once(path, old, new, 'c2 调用加 pbrule=', dry) is None:
                ok = False

    print('\n%s' % ('全部命中%s' % (' (dry-run)' if dry else ' (已写盘)') if ok else '有未命中项, 未写盘'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
