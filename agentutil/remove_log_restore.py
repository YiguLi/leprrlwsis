#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一次性重构：取消 `update_stats` 的"按通道还原成原始口径"。

为什么取消（本仓库实测，2026-10）
--------------------------------
`*_ddpg.py` 的 `record_update_stats` 分支在写 `update_stats` 之前，把 profit 字段
`×s_p`、cost 字段 `×s_c` 还原到原始口径，**但 `td_*` 不还原**（当时认为 TD 是
"自归一化"量，对 PER 无害 —— 这句对 PER 成立，对 CSV 不成立）。结果
`update_stats_H.csv` 变成**同一张表里混合量纲**：

    run96 末段 800 点实测（s_p=1000 / s_c=100）
      q_p_mean  = 3232.1    td_p_mean = 1395.9    td_p/q_p = 0.432  ← 同空间, 可信
      q_c_mean  =  228.9    td_c_mean =   28.57   td_c/q_c = 0.125  ← 真值 12.48 (差 100×)

`td_c/q_c` 被低估 100 倍，方向恰好是"看起来 cost critic 收敛得极好"，而它才是
相对 TD 误差更大的那条通道。`agentutil/diag_q_climb_cost.py` / `analyse_lam1.py` /
`diag_nolow_convergence.py` 都在算这个比值。

改动语义
--------
`update_stats` 的**所有**列一律留在**值函数回归空间**（= critic 实际回归、PER 实际
使用的那个空间），没有任何列再做 ×s。于是：

  * `td_* / q_*` 两个空间混用消失，比值恒有意义；
  * TensorBoard / CSV 的数字与网络实际输出一一对应，不再需要读者知道 s；
  * 跨 scale 比较 q 绝对值本来就没意义（D2 段断言 q_c 会随 s_c 线性变），
    真正 scale-free 的判据是比值 `td/|q|` 与 `λ_actor·q_c/q_p`。

**不动的东西**（这些不是日志约定，是物理口径）：
  * `λ` 区间 / `d_block` / EMA 对偶 / curriculum —— 全程原始口径；
  * `hsrl_agent_init` 的 bias 估计（读原始累加器、写网络时 ÷s）；
  * PER 优先级（`td_errors_*` 本就是未还原值）；
  * `reward_mean` 也在回归空间 —— 它与 `target_p_mean`、`min_p_next_mean` 三项
    同乘一个常数，`audit_policy_stability.check_gamma` 的一元回归 `a/g_eff` 不受影响。

用法
----
    python agentutil/remove_log_restore.py --check   # 只报告匹配情况
    python agentutil/remove_log_restore.py           # 执行
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))     # agentutil/
ROOT = os.path.dirname(_HERE)                          # 仓库根
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

# ---- 目标文件（含记录块的全部 agent）----
FILES = [
    'aifflor/aifflor_ddpg.py',
    'PER_ddpg.py',
    'prle/prle_ddpg.py',
    'singleaif/singleaif_ddpg.py',
    'td3leprrlrb/td3leprrlrb_ddpg.py',
    'td3per/td3per_ddpg.py',
    'td3perrb/td3perrb_ddpg.py',
    'td3perseg/td3perseg_ddpg.py',
    'td3prle/td3prle_ddpg.py',
    'td3prrbb/td3prrbb_ddpg.py',
    'td3prrlrb/td3prrlrb_ddpg.py',
    'yawflor/yawflor_ddpg.py',
]

# ---- 要删除的块：注释头 + 还原逻辑（全 profile 共用同一段文本）----
OLD_HEAD = (
    "            # 值函数口径字段还原成原始口径 (s=1 时恒等) —— update_stats_H.csv /\n"
    "            # analyse_q_scale.py / 历史 run 因此可直接对比。\n"
    "            if _SP != 1.0 or _SC != 1.0:\n"
    "                _d = self.update_stats[-1]\n"
)
OLD_PROFIT = (
    "                for _k in ('reward_mean', 'q_p_mean', 'q_p2_mean',\n"
    "                           'min_p_next_mean', 'target_p_mean'):\n"
    "                    if _k in _d:\n"
    "                        _d[_k] = float(_d[_k]) * _SP      # profit 通道用 s_p\n"
)
OLD_COST = (
    "                for _k in ('cost_mean', 'q_c_mean', 'q_c2_mean',\n"
    "                           'min_c_next_mean', 'target_c_mean'):\n"
    "                    if _k in _d:\n"
    "                        _d[_k] = float(_d[_k]) * _SC      # cost 通道用 s_c\n"
)

NEW_HEAD = (
    "            # ⚠ 本表**不还原**：所有列一律留在**值函数回归空间**（= critic 实际\n"
    "            # 回归、PER 实际使用的空间）。历史做法把 profit 字段 ×s_p、cost 字段\n"
    "            # ×s_c 还原成原始口径，但 `td_*` 不还原 ⇒ 同一张表里混合量纲，\n"
    "            # `td_c/q_c` 被低估 s_c 倍（run96: 0.125 vs 真值 12.48）。\n"
    "            # 跨 scale 需要的是**比值**判据（`td/|q|`、`λ_actor·q_c/q_p`），\n"
    "            # 它们在任一空间都成立；q 的绝对值本就随 s 线性变，比不得。\n"
)


def transform(text):
    """删除还原块；返回 (新文本, 命中类型)。"""
    n_head = text.count(OLD_HEAD)
    if n_head != 1:
        return None, 'HEAD x%d' % n_head
    out = text.replace(OLD_HEAD, NEW_HEAD)
    hit = []
    if OLD_PROFIT in out:
        out = out.replace(OLD_PROFIT, '')
        hit.append('profit')
    if OLD_COST in out:
        out = out.replace(OLD_COST, '')
        hit.append('cost')
    return out, '+'.join(hit) or 'NONE'


def main():
    check = '--check' in sys.argv
    bad = []
    for f in FILES:
        if not os.path.exists(f):
            bad.append((f, 'MISSING'))
            print('  %-34s 文件不存在' % f)
            continue
        with open(f, encoding='utf-8') as fh:
            src = fh.read()
        new, tag = transform(src)
        if new is None:
            bad.append((f, tag))
            print('  %-34s ✗ %s' % (f, tag))
            continue
        print('  %-34s ✓ 删除 %s（%d 字符）' % (f, tag, len(src) - len(new)))
        if not check:
            with open(f, 'w', encoding='utf-8', newline='') as fh:
                fh.write(new)
    if bad:
        print('\n✗ %d 个文件未处理：%s' % (len(bad), bad))
        sys.exit(1)
    print('\n✓ %d 个文件%s' % (len(FILES), '检查通过' if check else '已改写'))


if __name__ == '__main__':
    main()
