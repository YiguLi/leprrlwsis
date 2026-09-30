# -*- coding: utf-8 -*-
"""定位 aifflor 训练里 `PER_buffer.update_priorities` 的 `assert priority > 0` 失败。

已知：`new_priorities = |TD_p|/mean_p + |TD_c|/mean_c + PER_epsilon`（epsilon=0.01）
⇒ 优先级恒为正，**除非出现 NaN**（NaN>0 为 False）。所以根因必是 NaN。

本脚本只做**运行期插桩**（不改任何正字代码）：
  1. 包住 `SP_TD3.update_policy`，在返回前检查参与者是否有限
     （q/target/TD/priority/loss），首个非有限值就**抛异常中止** ⇒ 几秒内拿到现场；
  2. 包住 `PrioritizedReplayBuffer.update_priorities`，打印坏值样本；
  3. 打印首个非有限值**之前**最近几次 `update_policy` 的 q/td 量级轨迹，
     用于判断是"突然 NaN"还是"逐步发散"。

用法
----
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\diag_aifflor_nan.py --max_steps 5000
    ... --probe-after 20         # 前 20 次更新只记录不定罪（避免 warmup 期噪声）
"""
import argparse
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

from floris_import_guard import check_interpreter as _chk, sanitize as _sanitize  # noqa: E402
_chk()
_sanitize()

import numpy as np  # noqa: E402

ST = {'upd': 0, 'bad': 0, 'probe_after': 20,
      'name': {0: 'agent_H', 1: 'agent_L'}, 'hist': []}


class _Bad(Exception):
    pass


def _finite(name, val):
    """检查参与者是否有限；返回 (ok, 描述)。"""
    if val is None:
        return True, '%s=None' % name
    try:
        a = np.asarray(val, dtype=float)
    except Exception:
        return True, '%s=<non-numeric>' % name
    n_bad = int((~np.isfinite(a)).sum())
    if n_bad:
        return False, ('%s: %d/%d 非有限 (min=%r max=%r)'
                       % (name, n_bad, a.size,
                          float(np.nanmin(a)) if a.size else None,
                          float(np.nanmax(a)) if a.size else None))
    return True, '%s: ok (min=%+.4g max=%+.4g)' % (name, a.min(), a.max())


def install():
    import PER_buffer
    import aifflor.aifflor_ddpg as D

    # ---------- 1. update_priorities ----------
    _orig_up_pri = PER_buffer.PrioritizedReplayBuffer.update_priorities

    def spy_up_pri(self, idxes, priorities):
        pr = np.asarray(priorities, dtype=float)
        bad = ~(pr > 0)
        if bad.any():
            print('\n' + '!' * 92)
            print('!! update_priorities 收到非正优先级：坏 %d/%d' % (bad.sum(), pr.size))
            print('!! 坏值 : %s' % np.array2string(pr[bad][:8], precision=6))
            print('!! 全 NaN? %s | 含 inf? %s'
                  % (bool(np.isnan(pr).all()), bool(np.isinf(pr).any())))
            print('!' * 92)
        return _orig_up_pri(self, idxes, priorities)

    PER_buffer.PrioritizedReplayBuffer.update_priorities = spy_up_pri

    # ---------- 2. update_policy ----------
    _orig_pol = D.SP_TD3.update_policy

    def spy_pol(self, *a, **kw):
        ST['upd'] += 1
        who = ST['name'].get(id(self), 'agent?')
        # 记录入口前的 critic 统计（若已 NaN，说明是**上一步**留下的）
        snap = {}
        for attr in ('critic_p', 'critic_c', 'critic_p2', 'critic_c2',
                     'critic_target_p', 'critic_target_c'):
            net = getattr(self, attr, None)
            if net is None:
                continue
            try:
                w = next(net.parameters())
                snap[attr] = (float(w.detach().abs().max().item()),
                              float(w.detach().abs().mean().item()))
            except Exception:
                pass
        tgt = getattr(self, 'target_q_batch_p', None)
        out = _orig_pol(self, *a, **kw)

        if ST['upd'] < ST['probe_after']:
            return out
        # 出口检查
        checks = []
        for nm in ('q_batch_p', 'q_batch_c', 'target_q_batch_p', 'target_q_batch_c',
                   'TD_errors_p', 'TD_errors_c'):
            if hasattr(self, nm):
                checks.append(_finite(nm, getattr(self, nm)))
        if hasattr(self, 'td_mean_p'):
            checks.append(_finite('td_mean_p', self.td_mean_p))
            checks.append(_finite('td_mean_c', self.td_mean_c))
        bad = [msg for okk, msg in checks if not okk]
        if bad:
            ST['bad'] += 1
            print('\n' + '=' * 92)
            print('!! 首个非有限值 @ update_policy 第 %d 次 (%s)' % (ST['upd'], who))
            for m in bad:
                print('!!   %s' % m)
            print('!!   其余参与者也列出：')
            for okk, m in checks:
                if okk:
                    print('!!     %s' % m)
            print('!!   网络权重量级：')
            for k, (mx, mn) in snap.items():
                print('!!     %-18s |w|max=%.4g  |w|mean=%.4g' % (k, mx, mn))
            print('!!   最近 5 次历史：')
            for h in ST['hist'][-5:]:
                print('!!     %s' % h)
            print('=' * 92)
            traceback.print_stack()
            raise _Bad('update_policy 出现非有限值 (%s)' % who)
        ST['hist'].append(
            'upd#%d %s td_p_mean=%.4g td_c_mean=%.4g'
            % (ST['upd'], who, float(self.td_mean_p), float(self.td_mean_c)))
        return out

    D.SP_TD3.update_policy = spy_pol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_steps', type=int, default=5000)
    ap.add_argument('--probe-after', type=int, default=20,
                    help='前 N 次更新只记录不定罪（避开 warmup 期）')
    args, _u = ap.parse_known_args()
    ST['probe_after'] = args.probe_after

    install()
    import runpy
    sys.argv = [os.path.join(ROOT, 'aifflor', 'aifflor_main.py'),
                '--max_steps', str(args.max_steps)]
    try:
        runpy.run_module('aifflor.aifflor_main', run_name='__main__')
    except _Bad as e:
        print('\n[diag] 已定位：%s' % e)
        return 3
    except AssertionError:
        print('\n[diag] 原生 AssertionError（未被插桩捕获）')
        traceback.print_exc()
        return 3
    except SystemExit:
        pass
    print('\n[diag] 跑完 %d 步，update_policy %d 次，未发现非有限值'
          % (args.max_steps, ST['upd']))
    try:
        import aifflor.aifflor_turbine as AT
        s = AT.stats()
        print('[diag] FLORIS 层 NaN 处理统计:')
        print('       评估点总数 %d | 整批坏点 %d (%.4f%%) | 逐条重试救回 %d | 退回标称 %d'
              % (s['n_eval'], s['n_bad_in_batch'],
                 100.0 * s['n_bad_in_batch'] / max(s['n_eval'], 1),
                 s['n_retry_ok'], s['n_fallback']))
    except Exception as e:
        print('[diag] 统计读取失败: %s' % e)
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
