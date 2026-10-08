# -*- coding: utf-8 -*-
"""正式口径批量改写：warmup 期下层 = **价格条件随机充放**（`--warmup_bess price`，新默认）。

用户规格（2026-10-08）::

    风机行为范围 0.05–0.45（不变）、起点 1/3 → 0.2；SOC 起始 3（env 已是 3）；
    下层：300 电价随机向上充电 U(0.3,0.5)、580 不动、980 随机向下放电 U(0.3,0.5)；L 层逻辑不变。

本脚本在每个 main 的 `if step <= args.warmup:` 分支里接上新规则（**两种写法**）：

  A. 残差动作项目（td3per / td3perseg / td3prle / td3prrbb / aifflor / yawflor / singleaif）:
     在 `if warmup_rule is not None:` 之前插入 `getattr(warmup_rule,'is_price',False)` 分支
     —— `pb_cmd_L = warmup_rule.act(电价, pbrulemin, pbrulemax)`，并令
     `delta_L = [pb_cmd_L − pb_rule]`（残余语义不变，入池用 cmd/rule 核对，见 fill_bess_transition）。
  B. 影子电价项目（prle / prrbb）: 动作**就是**命令 ⇒ 只需把传给 `warmup_rule.action(...)` 的
     第一个参数在 price 臂下换成**电价**（`observation[0]`），其余不变。

`td3perrb / td3prrlrb / td3leprrlrb` **无 L 层**（只有 H 的 update_policy），不涉及。
`aifflorpave` 不进本表 —— 它由 `port_aifflorpave.py` 从改好的 aifflor_main 重新派生。

`--check`：磁盘版先反向还原、再正向改写，必须逐位回到磁盘版（往返证明）。

用法::

    python agentutil/port_warmup_bess_price.py            # dry-run
    python agentutil/port_warmup_bess_price.py --apply
    python agentutil/port_warmup_bess_price.py --check
    python agentutil/port_warmup_bess_price.py --revert
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

A_TARGETS = [
    'td3per/td3per_main.py',
    'td3perseg/td3perseg_main.py',
    'td3prle/td3prle_main.py',
    'td3prrbb/td3prrbb_main.py',
    'aifflor/aifflor_main.py',
    'yawflor/yawflor_main.py',
    'singleaif/singleaif_main.py',
]
B_TARGETS = [
    'prle/prle_main.py',
    'prrbb/prrbb_main.py',
]

A_OLD = ('            if warmup_rule is not None:\n'
         '                delta_L = np.zeros(1, dtype=np.float64)\n'
         '                pb_cmd_L = float(pb_rule)\n')
A_NEW = ('            if getattr(warmup_rule, \'is_price\', False):\n'
         '                # 新口径 (--warmup_bess price): 下层价格条件随机充放\n'
         '                #   300 低谷随机充电 +U(0.3,0.5) / 580 平峰不动 /\n'
         '                #   980 高峰随机放电 −U(0.3,0.5)，再夹进 RBB 可行窗口\n'
         '                #   (L 层其余逻辑不变: 残余动作 Δ = cmd − 规则基线)\n'
         '                pb_cmd_L = float(warmup_rule.act(float(observation[0]),\n'
         '                                                 rbb_info_obs[\'pbrulemin\'],\n'
         '                                                 rbb_info_obs[\'pbrulemax\']))\n'
         '                delta_L = np.array([pb_cmd_L - float(pb_rule)])\n'
         '            elif warmup_rule is not None:\n'
         '                delta_L = np.zeros(1, dtype=np.float64)\n'
         '                pb_cmd_L = float(pb_rule)\n')

B_OLD = ('                applied_action_L = original_action_L = warmup_rule.action(\n'
         '                    float(observation[2]), _w_obs["pbrulemin"], _w_obs["pbrulemax"])\n')
B_NEW = ('                # price 臂: 下层价格条件随机充放 (300 充 / 580 不动 / 980 放)\n'
         '                _x_warm = (float(observation[0])\n'
         '                          if getattr(warmup_rule, \'is_price\', False)\n'
         '                          else float(observation[2]))\n'
         '                applied_action_L = original_action_L = warmup_rule.action(\n'
         '                    _x_warm, _w_obs["pbrulemin"], _w_obs["pbrulemax"])\n')

TABLE = [(f, A_OLD, A_NEW, 'A') for f in A_TARGETS] + \
        [(f, B_OLD, B_NEW, 'B') for f in B_TARGETS]


def patch(text, path, log):
    for f, old, new, kind in TABLE:
        if f != path:
            continue
        n = text.count(old)
        log.append(('%s :: %s 形状' % (path, kind), n))
        if n != 1:
            return None, log
        return text.replace(old, new, 1), log
    log.append(('%s :: 未登记' % path, 0))
    return None, log


def revert(text, path, log):
    for f, old, new, kind in TABLE:
        if f != path:
            continue
        n = text.count(new)
        log.append(('%s :: 反向 %s' % (path, kind), n))
        return (text.replace(new, old, 1) if n == 1 else None), log
    log.append(('%s :: 未登记' % path, 0))
    return None, log


def main():
    apply = '--apply' in sys.argv
    check = '--check' in sys.argv
    do_revert = '--revert' in sys.argv
    paths = [f for f, _o, _n, _k in TABLE]
    ok = True
    if do_revert:
        for p in paths:
            disk = open(p, encoding='utf-8').read()
            rev, _ = revert(disk, p, [])
            if rev is None:
                print('  %-34s **反向失败（未命中）**' % p)
                ok = False
                continue
            open(p, 'w', encoding='utf-8', newline='').write(rev)
            print('  %-34s 已回滚' % p)
        return 0 if ok else 1
    for p in paths:
        disk = open(p, encoding='utf-8').read()
        if check:
            rev, _ = revert(disk, p, [])
            if rev is None:
                print('  %-34s **反向未命中**' % p)
                ok = False
                continue
            rep, log = patch(rev, p, [])
            good = (rep == disk)
            ok = ok and good
            print('  %-34s %s  往返逐位一致=%s' % (p, 'OK  ' if good else '**失败**', good))
            continue
        new, log = patch(disk, p, [])
        good = new is not None
        ok = ok and good
        print('  %-34s %s  %s' % (p, 'OK  ' if good else '**失败**',
                                  '  '.join('%s=%d' % (t.split(':: ')[-1], n) for t, n in log)))
        if good and apply:
            open(p, 'w', encoding='utf-8', newline='').write(new)
    print(('--check: 全部可逆 ✅' if ok else '--check: **FAIL**') if check else
          ('✅ 已写盘 %d 个 main（aifflorpave 需另行重新派生）' % len(paths) if apply
           else '(dry-run)'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
