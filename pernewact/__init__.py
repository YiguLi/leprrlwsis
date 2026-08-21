# pernewact — PER + 有界行为输出（clamp + straight-through）
#
# 基于 PER 项目（PER_main0.py / PER_ddpg.py / myfunc.py）修改：
#   - 风机轴向因子输出改为 clamp + straight-through（无 sigmoid/tanh，无梯度饱和）
#   - 行为恒在 [a_lo, a_hi] 内，不存在越界（风机维），env/critic/target 只见界内动作
#   - 记录项记录 clamp 后最终值；tensorboard 记录上层 Qp/Qc 均值
