# PRRBB (Wind × Price Segmented Reward + PER lower baseline) Project
# =============================================================================
# Upper layer (H): Decomposes PG revenue into turbine (ptur) and BESS (pb)
# components, each with wind-speed × price-level joint segmentation coefficients.
#
#   r1 = w_wind * (pr * ptur * time)       ← turbine profit, 9-case segmented
#      + w_bess * (-pr * pbrule * time)    ← BESS profit, 9-case segmented
#      - 490.0 * max(-pbrule, 0) * time    ← degradation, NO coefficient
#      - penalty_NN_H
#
#   Segmentation matrix (wind_level, price_level) → (w_wind, w_bess):
#     LOW wind  (<7.0):   (1.2,0.8) (1.2,1.0) (1.2,1.2)
#     MED wind ([7.0,11.0)): (1.0,0.8) (1.0,1.0) (1.2,1.2)
#     HIGH wind (>=11.0):  (0.7,0.8) (0.8,1.0) (0.9,1.2)
#
# Lower layer (L): PER baseline — NO segmentation.
#   r2 = 10.0 * shadow_profit - 300.0 * soc_deviation
#   soc_deviation: standard [1.2, 4.8] safe range only
#
# Philosophy: Wind farm maximizes profit under safety constraints.
# Low risk (low wind) → aggressive generation.
# Medium risk → normal operation.
# High risk (high wind) → cautious profit-seeking.
# BESS safety is the lower layer's responsibility; upper layer only
# adjusts BESS profit weights to steer shadow-price regulation.
#
# Shadow price signal (PER convention): range (-0.5, 0.5), kept unchanged.
# RBB clipping: same hard clipping before env.step (unchanged).
