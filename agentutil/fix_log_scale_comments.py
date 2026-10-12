#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""清理首批改动后残留的**过期注释**（纯注释/文案，不动任何数值）。

三处：
  1. `TD_errors_* = ... * _SP   # ×s_p   # ×s: PER/日志保持原始口径`
     —— 这句的后半在新口径下是错的：PER 的优先级是
     `|TD|·s / mean(|TD|·s)`，分子分母同乘 s ⇒ 本就与 s 无关；而"日志"现在
     一律留回归空间，不需要 ×s。乘以 s 的**唯一**效果是让 `_SC` 对 `_SD`
     配平（避免一个通道用步级、一个用块级混进同一条优先级）。
  2. `lam_loss = (min_c_cur * _SC - self.cost_limit)  # 还原原始口径 (仅 non-EMA 分支)`
     —— 措辞改为"非线性对偶分支"，与 `--c2-mode`/`c1-mode` 无关，避免与"还原"混淆。
  3. 记录块头部注释改为**不含"还原"二字**的措辞 —— 便于 `grep 还原` 只命中真问题。
"""
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

AGENTS = [
    'aifflor/aifflor_ddpg.py', 'PER_ddpg.py', 'prle/prle_ddpg.py',
    'singleaif/singleaif_ddpg.py', 'td3leprrlrb/td3leprrlrb_ddpg.py',
    'td3per/td3per_ddpg.py', 'td3perrb/td3perrb_ddpg.py',
    'td3perseg/td3perseg_ddpg.py', 'td3prle/td3prle_ddpg.py',
    'td3prrbb/td3prrbb_ddpg.py', 'td3prrlrb/td3prrlrb_ddpg.py',
    'yawflor/yawflor_ddpg.py',
]

# 1) TD 误差注释：吃掉原注释整段（含 PER_ddpg 的重复尾巴；cost 侧无尾巴）
TD_RE = re.compile(
    r"\) \* (_S[PC])\s+#\s*×s_[pc](\s*:\s*PER/日志保持原始口径)?(\s+#[^\n]*)?")
TD_NEW = {
    '_SP': ") * _SP   # 仅用于 PER 配平 (分子分母同乘 s ⇒ 优先级与 s 无关); 日志一律留回归空间",
    '_SC': ") * _SC   # 仅用于 PER 配平 (同上); 日志一律留回归空间",
}

# 2) lam_loss 注释
LAM_OLD = "# 还原原始口径 (仅 non-EMA 分支)"
LAM_NEW = "# 非线性对偶分支 (仅覆盖层/单步模式)"

# 3) 记录块头部：去掉"还原"字样
HDR_OLD = (
    "            # ⚠ 本表**不还原**：所有列一律留在**值函数回归空间**（= critic 实际\n"
    "            # 回归、PER 实际使用的空间）。历史做法把 profit 字段 ×s_p、cost 字段\n"
    "            # ×s_c 还原成原始口径，但 `td_*` 不还原 ⇒ 同一张表里混合量纲，\n"
    "            # `td_c/q_c` 被低估 s_c 倍（run96: 0.125 vs 真值 12.48）。\n"
    "            # 跨 scale 需要的是**比值**判据（`td/|q|`、`λ_actor·q_c/q_p`），\n"
    "            # 它们在任一空间都成立；q 的绝对值本就随 s 线性变，比不得。\n"
)
HDR_NEW = (
    "            # ⚠ 本表的**每一列都在值函数回归空间**（= critic 实际回归、PER 实际\n"
    "            # 使用的空间），不做任何 ×s 换算。历史做法把 profit 字段 ×s_p、cost\n"
    "            # 字段 ×s_c 换成原始口径，而 `td_*` 留在回归空间 ⇒ 同一张表里混了两种\n"
    "            # 量纲，`td_c/q_c` 因此被低估 s_c 倍（run96: 0.125 vs 真值 12.48），\n"
    "            # 且方向是「看起来 cost critic 收敛得极好」——与事实相反。\n"
    "            # 跨 scale 需要的是**比值**判据（`td/|q|`、`λ_actor·q_c/q_p`）：\n"
    "            # 它们在任一空间都成立，而 q 的绝对值本就随 s 线性变，比不得。\n"
)


def main():
    check = '--check' in sys.argv
    n_td = n_lam = n_hdr = 0
    bad = []
    for f in AGENTS:
        with open(f, encoding='utf-8') as fh:
            src = fh.read()
        out, k = TD_RE.subn(lambda m: TD_NEW[m.group(1)], src)
        k2 = out.count(LAM_OLD)
        out = out.replace(LAM_OLD, LAM_NEW)
        k3 = out.count(HDR_OLD)
        out = out.replace(HDR_OLD, HDR_NEW)
        n_td += k
        n_lam += k2
        n_hdr += k3
        # 每个 agent 期望：TD 2 处（profit/cost 各一）、lam 1 处、头部 1 处
        if (k, k2, k3) != (2, 1, 1):
            bad.append((f, k, k2, k3))
        if out != src and not check:
            with open(f, 'w', encoding='utf-8', newline='') as fh:
                fh.write(out)
    print('TD 注释修正 %d 处 / lam_loss 注释 %d 处 / 记录块头部 %d 处 %s'
          % (n_td, n_lam, n_hdr, '（检查模式，未写入）' if check else '（已写入）'))
    if bad:
        print('✗ 以下文件命中数不符（期望每文件 2/1/1）：')
        for b in bad:
            print('   %-34s TD=%d lam=%d hdr=%d' % b)
        sys.exit(1)
    print('✓ 12 个 agent 全部符合 2/1/1')


if __name__ == '__main__':
    main()
