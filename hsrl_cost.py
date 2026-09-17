"""
HSRL 统一 H 层约束口径 —— 波动当量（单一真源）
==============================================
本模块是所有带 c1 的项目共享的**唯一数字来源**，取代此前各项目各写一份、
互不一致的标定（实测：gap 系数 50 vs 10、d_block_H 1.2 / 12 / 50、
λ1 区间 20/3、50/3、100/3、500/50 —— 同一物理事件被定了不同的价）。

设计（见对话记录）
------------------
1. **c1 统一为"波动当量" [MW·step]**
       c1 = P_F + K_V · max(0, P_F − 3)
   * 平滑项斜率 1（不再被 0.1 的斜率淹没 ⇒ λ_min 能真正承载"平抑"偏好）
   * 越限项额外 K_V 倍（K_V = 20 由"单次典型事件 0.85 MW 吃掉整块额度"反推）
   * `unfeasible_gap` 不再单列：它与"越限"是同一事件（实测触发率逐位相同 0.056%）
2. **预算 d_block ≡ D_BLOCK = 3 MW × 5 步 = 15**
   * 语义 = 硬约束允许的最大消耗（"贴着 3 MW 上限跑满整块"）
   * 地板 = 初值 ⇒ 取消 curriculum（对事件型约束收紧预算只会与硬约束冲突）
3. **λ**：λ_min = RHO_S·PR_REF·DT ≈ 0.967 元/MW·step（= 平抑支付意愿，占边际收益 10%）
       λ_max = 5·λ_min，初值 = λ_min；乘性对偶 λ ← λ·exp(η·(EMA/d − 1))
4. **追踪（块末 SOC 目标残差）移出 c1 → 进 r1**，固定价 W_GAP（元/MWh）
   * 偏好用固定价，约束才配对偶；避免 λ 同时定价追踪
5. **--c1-mode uniform|legacy**：uniform 走本模块；legacy 保留各项目原公式（对照臂）

用法（各项目一处）
------------------
    import hsrl_cost as hc                 # main: hc.set_mode(args.c1_mode)
    c1 = hc.c1_apply(c1_legacy, P_F)       # myfunc: uniform 时替换为波动当量
    ...  _h = hc.h_settings()              # main: d_block / λ 区间 / 初值
    ...  hc.dual_update(agent_H, d_block_H) # main: 代替 agent.update_lam_ema
"""
import numpy as np

# ---------------------------------------------------------------- 恒等常量
MODE = 'uniform'          # 'uniform' | 'legacy'（由 main 的 --c1-mode 设定）

DT = 1.0 / 60.0           # 步长 (h)
P_F_LIMIT = 3.0           # 波动硬限 (MW)
K_V = 20.0                # 越限额外倍率（越限边际斜率 = 1 + K_V = 21）
D_BLOCK = 15.0            # = 3 MW × 5 步 = 硬约束允许的最大块消耗 [MW·step]

RHO_S = 0.10              # λ_min / (pr_ref·dt)：平抑支付意愿（边际收益的 10%）
PR_REF = 580.0            # 名义电价 (MED)
LAM_MIN = RHO_S * PR_REF * DT      # ≈ 0.9667 元/MW·step
LAM_MAX = 5.0 * LAM_MIN            # ≈ 4.833
LAM_INIT = LAM_MIN
ETA_LAM = 0.02            # 乘性对偶步长
ALPHA_EMA = 0.02          # 块成本 EMA 系数（与各 main 一致）

W_GAP = 10.0              # 追踪偏好固定价 (元/MWh)，进 r1
SOC_TOL = 0.1             # 块末残差豁免 (MWh)


# ---------------------------------------------------------------- 模式开关
def set_mode(m):
    """由 main 从 --c1-mode 调用；任何非 'uniform' 值都落到 legacy。"""
    global MODE
    MODE = 'uniform' if str(m).strip().lower() == 'uniform' else 'legacy'
    return MODE


def mode():
    return MODE


# ---------------------------------------------------------------- 成本 / 偏好
def fluct_cost(P_F):
    """波动当量 [MW·step]：c1 = P_F + K_V·(P_F−3)⁺。"""
    p = float(P_F)
    return p + K_V * max(0.0, p - P_F_LIMIT)


def c1_apply(c1_legacy, P_F):
    """uniform: 用波动当量替换 legacy c1；legacy: 原样返回。"""
    return fluct_cost(P_F) if MODE == 'uniform' else c1_legacy


def gap_pref(residual, tol=SOC_TOL):
    """追踪偏好（元，进 r1）：W_GAP·max(0, |δ| − tol)。"""
    return W_GAP * max(0.0, abs(float(residual)) - float(tol))


# ---------------------------------------------------------------- 对偶设置 / 更新
def h_settings(m=None):
    """uniform 时返回 H 层统一设置；legacy 时返回 None（由各项目自用原值）。"""
    if (m or MODE) != 'uniform':
        return None
    return dict(d_block=D_BLOCK,
                d_block_min=D_BLOCK,
                lam_min=LAM_MIN,
                lam_max=LAM_MAX,
                lam_init=LAM_INIT,
                ema_lam_lr=ETA_LAM,
                gap_in_r=True)


def dual_update(agent, d_block):
    """H 层对偶上升：uniform=乘性（对 EMA/d 的比值），legacy=原加性。

    乘性形式对尺度免疫（λ 是"罚款倍率"），避免 legacy 加性式在 c 换量纲后顶格。
    """
    if agent is None or getattr(agent, 'ema_cost', None) is None:
        return
    if MODE == 'uniform':
        ratio = float(agent.ema_cost) / max(float(d_block), 1e-9)
        agent.lam *= float(np.exp(float(agent.ema_lam_lr) * (ratio - 1.0)))
    else:
        agent.lam += float(agent.ema_lam_lr) * (float(agent.ema_cost) - float(d_block))
    agent.lam = max(float(agent.lam), float(agent.lam_min))
    if getattr(agent, 'lam_max', None) is not None:
        agent.lam = min(float(agent.lam), float(agent.lam_max))


# ---------------------------------------------------------------- 诊断 / 打印
def summary(pr=None):
    """一行口径说明（startup 打印用）。"""
    pr = float(PR_REF if pr is None else pr)
    marg = pr * DT                                   # 边际收益 元/(MW·step)
    smooth_mul = LAM_MIN / marg
    viol_mul = (1.0 + K_V) * LAM_MIN / marg
    return ('[c1-mode={}] c1 = P_F + {:.0f}·(P_F−3)+ [MW·step]; d_block = {:.0f} '
            '(= 3MW×5步); lam∈[{:.3f},{:.3f}] 初值 {:.3f}; '
            '有效价: 平滑 λ={:.3f}元/MW·step (={:.0%} 边际收益), '
            '越限 {:.1f}元/MW·step (={:.1f}× 边际收益)'
            .format(MODE, K_V, D_BLOCK, LAM_MIN, LAM_MAX, LAM_INIT,
                    LAM_MIN, smooth_mul, (1.0 + K_V) * LAM_MIN, viol_mul))
