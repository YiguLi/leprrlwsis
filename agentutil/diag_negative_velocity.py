# -*- coding: utf-8 -*-
"""追查 `Some velocities at the rotor are negative` 警告的真来源。

警告出自 `floris_model.py:551`：
    if (self.core.flow_field.u < 0.0).any():
        self.logger.warning("Some velocities at the rotor are negative.")
⚠ 注意它检查的是**整个流场网格 `flow_field.u`**（`(findex, nt, gz, gy)`），
**不只是转子点** —— 所以消息用词有误导性。

本脚本直接检查流场，回答：
  ① 负速度出现在哪个网格位置（转子中心？叶尖？轴向哪个位置？）；
  ② 是转子处的 inflow 还是**别的机组的尾流叠加区**；
  ③ 与标定参数（窄尾流 alpha=0.1446 / ka=0.05）是否相关（对照 stock 参数）；
  ④ 是否与风速有关。

用法（不需 MATLAB）
------------------
    F:\\anaconda\\envs\\wsis\\python.exe agentutil\\diag_negative_velocity.py
"""
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


def probe(tag, m, us, A, note=''):
    R._A_SET = np.asarray(A, dtype=float)
    n = len(us)
    R._run_batch(m, us, A)                       # 触发 run()，填好 flow_field
    ff = m.core.flow_field
    u = np.asarray(ff.u, dtype=float)
    neg = u < 0.0
    print('-' * 84)
    print('%s  %s' % (tag, note))
    print('  flow_field.u 形状 %s | 最小 %.6f | 负值点数 %d / %d'
          % (np.shape(u), float(u.min()), int(neg.sum()), u.size))
    if not neg.any():
        print('  ⇒ 无负速度')
        return
    # 定位
    idx = np.argwhere(neg)
    for k in range(min(5, len(idx))):
        f, t, gz, gy = idx[k]
        print('     第 %d 例: findex=%d 机组=%d 网格(z,y)=(%d,%d)  u=%.6f'
              % (k + 1, f, t, gz, gy, u[f, t, gz, gy]))
    # 按机组统计
    per_t = neg.sum(axis=(0, 2, 3))
    print('     按机组: %s' % np.array2string(per_t, precision=0))
    # 网格点位置（转子中心索引）
    g = m.core.grid
    if hasattr(g, 'z_sorted'):
        zc = g.z_sorted[:, :, g.z_sorted.shape[2] // 2, g.z_sorted.shape[3] // 2]
        print('     转子中心高度 z = %s' % np.array2string(np.ravel(zc)[:5], precision=2))
    # 负值点的 (z, y) 相对转子中心
    try:
        zs = np.asarray(g.z_sorted, dtype=float)
        ys = np.asarray(g.y_sorted, dtype=float)
        for k in range(min(3, len(idx))):
            f, t, gz, gy = idx[k]
            print('       → 该点 (z,y)=(%.2f, %.2f)  vs 转子中心 (%.2f, %.2f)'
                  % (zs[f, t, gz, gy], ys[f, t, gz, gy],
                     zs[f, t, zs.shape[2] // 2, zs.shape[3] // 2],
                     ys[f, t, ys.shape[2] // 2, ys.shape[3] // 2]))
    except Exception as e:
        print('     (网格坐标读取失败: %s)' % e)


def main():
    print('=' * 84)
    print('追查负速度来源')
    print('=' * 84)
    us = np.array([10.0])
    A = np.full((1, 5), 1.0 / 3.0)

    m_cal, info_cal = AT.build_model(calibrated=True)
    print('\n[标定参数] θ = %s' % dict((('%s.%s' % k), round(v, 5))
                                      for k, v in AT.CALIBRATED_WAKE_PARAMS.items()))
    probe('标定参数 / u=10 / a=1/3', m_cal, us, A)

    m_stock, _ = AT.build_model(calibrated=False)
    probe('随包默认 / u=10 / a=1/3', m_stock, us, A)

    # 扫风速，看负值何时出现
    print('\n' + '=' * 84)
    print('负值是否随风速出现（标定参数，a=1/3，每次单独 run）')
    print('=' * 84)
    print('  %-8s %12s %10s' % ('u', 'min(u_field)', '负值点数'))
    for uu in (4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0):
        R._A_SET = A
        R._run_batch(m_cal, np.array([uu]), A)
        uf = np.asarray(m_cal.core.flow_field.u, dtype=float)
        print('  %-8.1f %12.6f %10d' % (uu, float(uf.min()), int((uf < 0).sum())))

    # 扫动作，看是否与 a 有关
    print('\n' + '=' * 84)
    print('负值是否随动作出现（标定参数，u=10）')
    print('=' * 84)
    print('  %-14s %12s %10s' % ('动作', 'min(u_field)', '负值点数'))
    for tag, aa in (('全场 0.025', 0.025), ('全场 0.05', 0.05), ('全场 0.2', 0.2),
                    ('全场 1/3', 1 / 3), ('全场 0.45', 0.45), ('全场 0.5', 0.5)):
        AA = np.full((1, 5), aa)
        R._A_SET = AA
        R._run_batch(m_cal, np.array([10.0]), AA)
        uf = np.asarray(m_cal.core.flow_field.u, dtype=float)
        print('  %-14s %12.6f %10d' % (tag, float(uf.min()), int((uf < 0).sum())))
    return 0


if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass
    sys.exit(main())
