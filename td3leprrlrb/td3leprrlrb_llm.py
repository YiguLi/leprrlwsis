"""
TD3LEPRRLRB LLM Analysis Module
===============================
LLM 元控制器: 依据一个调整窗口 (默认 7200 步 = 5 天) 的运行统计, 微调
【风速分段阈值】(WIND_LOW_MAX / WIND_HIGH_MIN), 也就是本项目里唯一被 LLM
触碰的旋钮。

两个组件:
  1. WindowStats            —— 窗口内逐步指标累积 (r1 / 利润 / 越限 / 不可行 /
                               块末 SOC 目标罚 / 块末 SOC 残差 / 各风速档统计)
  2. llm_analyze_thresholds —— 组织提示词, 调用 LLM, 解析出
                               (ΔLOW_MAX, ΔHIGH_MIN)

设计要点 (与 td3prle 的继承关系)
--------------------------------
* 结构与 td3prle/td3prle_llm.py 一致 (WindowStats + 提示词 + JSON 解析 +
  provider 工厂); 差别在于本项目的下层是【规则控制器】而不是 RL, 因此:
    - 快照里不再有 r2/c2/裁剪量, 而是新增【块末 SOC 目标罚 gap_pen】与
      【块末 SOC 残差 soc_track_err】—— 它们刻画"H 下达的目标是否被下层
      在 5 步块内做到", 是规则下层的核心健康指标;
    - 额外给出真实经济口径 profit_avg (P_G·pr·dt − 退化), 而不只是 r1 代理。
* LLM 调用失败 / 未接 provider 时一律退化为占位模式 (返回 (0.0, 0.0)),
  训练继续 —— 阈值保持不动, 等价于退回 td3prrlrb 的行为。
"""

import json
import os


class WindowStats:
    """Accumulates per-step training metrics over one adjustment window.

    Tracks reward / profit totals, violation & infeasibility counts, the
    block-end SOC-target penalty (规则下层特有), and per-wind-level
    statistics, to give the LLM a complete picture of how the agent behaves
    under the CURRENT thresholds.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all accumulators for a new window."""
        self.step_count = 0
        self.r1_total = 0.0
        self.profit_total = 0.0
        self.violation_count = 0          # steps where P_F > 3.0
        self.infeasible_count = 0         # RBB infeasible events (窗口为空)
        # ---- 块末 SOC 目标可行性 (每 5 步结算一次) ----
        self.block_count = 0
        self.gap_pen_total = 0.0          # Σ 块末 gap 罚
        self.gap_pen_blocks = 0           # gap 罚 > 0 的块数
        self.soc_err_total = 0.0          # Σ 块末 |soc_target − SOC|
        # ---- 分层统计: [steps, r1, profit, violations] ----
        self.level = {k: {'steps': 0, 'r1': 0.0, 'profit': 0.0, 'viol': 0}
                      for k in ('LOW', 'MEDIUM', 'HIGH')}

    # ------------------------------------------------------------------
    def add(self, r1, wind_level, P_F, is_infeasible, profit=0.0):
        """Add one step's metrics.

        Parameters
        ----------
        r1 : float         上层奖励 (H 层 r1)
        wind_level : str   'LOW' | 'MEDIUM' | 'HIGH' (按【当前】阈值分档)
        P_F : float        并网功率波动 |P_G(t) − P_G(t−1)| (MW)
        is_infeasible : bool  RBB 窗口为空 (物理+波动约束无交集)
        profit : float     本步真实利润 P_G·pr·dt − 退化 (RMB)
        """
        self.step_count += 1
        self.r1_total += float(r1)
        self.profit_total += float(profit)

        lv = self.level.get(wind_level, self.level['MEDIUM'])
        lv['steps'] += 1
        lv['r1'] += float(r1)
        lv['profit'] += float(profit)

        if P_F > 3.0 + 1e-4:
            self.violation_count += 1
            lv['viol'] += 1
        if is_infeasible:
            self.infeasible_count += 1

    def add_block_end(self, gap_pen, soc_err):
        """Add one 5-step block's end-of-block SOC-target bookkeeping.

        gap_pen : float   块末 SOC 目标可行性罚 = W_GAP·max(0, |target−SOC|−tol)
        soc_err : float   块末残差 |soc_target − SOC_end| (MWh)
        """
        self.block_count += 1
        self.gap_pen_total += float(gap_pen)
        self.soc_err_total += float(soc_err)
        if gap_pen > 0.0:
            self.gap_pen_blocks += 1

    # ------------------------------------------------------------------
    def summary(self):
        """Return the window summary dict consumed by the prompt builder."""
        n = max(self.step_count, 1)
        nb = max(self.block_count, 1)
        out = {
            'total_steps': self.step_count,
            'blocks': self.block_count,
            'r1_total': round(self.r1_total, 2),
            'r1_avg': round(self.r1_total / n, 4),
            'profit_total': round(self.profit_total, 2),
            'profit_avg': round(self.profit_total / n, 4),
            'violation_rate': round(self.violation_count / n, 4),
            'infeasible_count': self.infeasible_count,
            'gap_pen_total': round(self.gap_pen_total, 3),
            'gap_pen_per_block': round(self.gap_pen_total / nb, 4),
            'gap_pen_block_rate': round(self.gap_pen_blocks / nb, 4),
            'soc_track_err_avg': round(self.soc_err_total / nb, 4),
        }
        for key, name in (('HIGH', 'high_wind'), ('MEDIUM', 'med_wind'),
                          ('LOW', 'low_wind')):
            lv = self.level[key]
            m = max(lv['steps'], 1)
            out[name] = {
                'steps': lv['steps'],
                'pct': round(100.0 * lv['steps'] / n, 1),
                'r1_avg': round(lv['r1'] / m, 4),
                'profit_avg': round(lv['profit'] / m, 4),
                'violation_rate': round(lv['viol'] / m, 4),
            }
        return out


# =============================================================================
# Prompt template (中文; 输出严格 JSON)
# =============================================================================

_PROMPT_TEMPLATE = """你是一个风储混合发电系统的风险调度专家。系统包含5台风机和1组锂电池储能（BESS），
上层用强化学习智能体（SP-TD3）在电力现货市场中做实时调度：它每一步输出5台风机的桨距
和1个「SOC目标信号」，下层是一套【确定性规则控制器】——在电网波动约束允许的功率窗口内，
用最大能力把电池SOC拉向上层给出的目标。

电网要求相邻时刻的并网功率波动不超过3.0 MW，否则记为一次越限违规。
上层对「下层做不到的SOC目标」付费（块末结算），因此上层必须给出物理可达的目标。

═══════════════════════════════════════════
当前风险分段阈值（你唯一可以微调的旋钮）
═══════════════════════════════════════════
低风速（LOW）：  u < {current_low_max} m/s
中风速（MEDIUM）：{current_low_max} ≤ u < {current_high_min} m/s
高风速（HIGH）：  u ≥ {current_high_min} m/s

【阈值如何影响学习】风速档同时决定两件事：
  (1) 收益权重：风机收益乘 WT_WIND[档]（LOW=1.2 鼓励多发 / MEDIUM=1.0 / HIGH=0.9 抑制激进）；
      储能（BESS）收益权重只由电价档决定，与风速档、SOC 位置均无关。
  (2) 风险系数：波动成本乘 C1_RISK[档]（LOW=0.8 / MEDIUM=1.0 / HIGH=1.2）。
所以：
  - 调高 HIGH_MIN → 更少样本归入HIGH → 风机收益权重抬升、波动风险系数下降
    → 整体盈利空间增大，但风险敞口也增大；
  - 调低 HIGH_MIN → 更多样本归入HIGH → 更保守，盈利空间缩小，但更安全；
  - 调高 LOW_MAX → 更多样本归入LOW（收益×1.2、风险×0.8）→ 中等风速也被鼓励多发；
  - 调低 LOW_MAX → 该区间趋于保守。
  注意：SOC 的约束不通过分段权重实现，而是由「上层给定SOC目标 → 下层规则跟踪 →
  上层在块末为做不到的目标付费」这条链路单独承担，因此调整阈值不是控制 SOC 的手段。

【调整范围硬约束（重要）】
训练开始时的原设定阈值：LOW_MAX = {orig_low_max} m/s，HIGH_MIN = {orig_high_min} m/s。
整个训练过程中，两个阈值只能在 原设定 ±1 m/s 内变动：
  LOW_MAX  ∈ [{adj_low_min}, {adj_low_max}] m/s
  HIGH_MIN ∈ [{adj_high_min}, {adj_high_max}] m/s
你的建议是「对当前值的增量」；若增量使阈值越出上述范围，系统会将它截断到边界。
请务必直接给出带内可行的数值，不要建议越出该调整带。

═══════════════════════════════════════════
当前窗口统计数据（{total_steps}步 = {blocks}个5步块）
═══════════════════════════════════════════
平均r1（加权盈利代理）：{r1_avg}
平均真实利润 profit（= P_G·电价·dt − 退化成本）：{profit_avg}
整体越限率：{violation_rate}（越限 = |ΔP_G| > 3.0 MW）
不可行事件数：{infeasible_count}（物理约束与波动约束无交集，违规物理上无法避免）

【规则下层的目标跟踪质量（本项目特有）】
块末SOC目标罚合计：{gap_pen_total}，平均每块 {gap_pen_per_block}，
  出现惩罚的块占比：{gap_pen_block_rate}
块末平均 |目标SOC − 实际SOC|：{soc_track_err_avg} MWh
（该罚只惩罚「上层给出的目标在5步块内做不到」的部分。它不由风速阈值直接决定，
 但数值持续偏高说明上层在过高/过低的SOC目标上反复试探，属于系统风险信号。）

按风速分段（按【当前】阈值分档）：
  HIGH   风速：{high_steps}步（{high_pct}%），平均r1 = {high_r1_avg}，
         平均profit = {high_profit_avg}，越限率 = {high_violation_rate}
  MEDIUM 风速：{med_steps}步（{med_pct}%），平均r1 = {med_r1_avg}，
         平均profit = {med_profit_avg}
  LOW    风速：{low_steps}步（{low_pct}%），平均r1 = {low_r1_avg}，
         平均profit = {low_profit_avg}

═══════════════════════════════════════════
历史调整记录（最近5次）
═══════════════════════════════════════════
{history_text}

═══════════════════════════════════════════
决策指南
═══════════════════════════════════════════
你需要根据以上数据判断：当前的两个风速阈值应该向上调、向下调、还是保持不变。

分析框架（按优先级排序）：

1.【安全检查】HIGH风速越限率是否过高（>5%）？
  → 是：下调HIGH_MIN（扩大HIGH区，把更多样本纳入高风险系数档），即使牺牲盈利也要控风险。
     建议幅度 -0.3~-0.5。
  → 否：进入第2步。

2.【盈利优化】整体越限率是否很低（<1%）且HIGH越限率也低（<2%）？
  → 是：系统当前过于保守，可以上调HIGH_MIN（缩小HIGH区）释放盈利空间。建议幅度 +0.1~+0.3。
  → 否：进入第3步。

3.【不可行事件】infeasible_count 是否持续增长？
  → 是：风速已超出系统的物理调节能力，不应继续上调HIGH_MIN；
     同时可考虑下调HIGH_MIN，提前抑制高风险动作。

4.【LOW/MEDIUM边界微调】LOW与MEDIUM的权重差异主要在储能端与风险系数（0.8 vs 1.0）。
  除非LOW风速占比过大导致过度激励，否则LOW_MAX通常无需大幅调整。
  建议将LOW_MAX的调整幅度控制在 ±0.1~±0.2。

5.【目标跟踪信号】若 gap_pen_per_block 明显偏高且越限率也高，
  说明系统同时面临「目标不可达」与「波动风险」，此时优先保守：下调HIGH_MIN；
  若 gap_pen 高但越限率很低，则阈值调整空间不大，建议本次维持不变。

6.【振荡抑制】检查历史记录：如果上一次调整方向与本次相反，说明系统在震荡。
  → 将本次调整幅度减半（阻尼），或维持不变。

7.【幅度约束】单次调整幅度不得超过 ±0.5 m/s。优先采用小幅渐进策略（±0.1~±0.2），
  仅在越限率显著超标（>8%）时使用较大幅度（±0.3~±0.5）。

8.【全局硬约束】结合上文【调整范围硬约束】：任何建议不得使 LOW_MAX / HIGH_MIN 越出
  原设定 ±1 m/s 范围（越界部分会被系统截断）。请结合历史调整与当前值在带内渐进寻优。

═══════════════════════════════════════════
输出格式（严格遵守）
═══════════════════════════════════════════
只输出一行合法的JSON，不要包含```json```代码块标记，不要包含任何其他文字。
JSON格式如下：
{{"adjust_low_max": <float>, "adjust_high_min": <float>, "reasoning": "<一句话理由>"}}

示例输出：
{{"adjust_low_max": 0.0, "adjust_high_min": -0.3, "reasoning": "HIGH风速越限率达7.2%，超过5%安全线，下调HIGH_MIN以扩大高风速保守区，降低违规风险"}}"""


def _fmt_history(history):
    """Render the last-5 adjustment records as prompt text."""
    lines = []
    for i, h in enumerate(history[-5:]):
        lines.append(
            "  窗口{}: Step {}, LOW_MAX {:.1f}→{:.1f}, HIGH_MIN {:.1f}→{:.1f}".format(
                i + 1, h.get('step', '?'),
                h['old_low_max'], h['new_low_max'],
                h['old_high_min'], h['new_high_min']))
    return "\n".join(lines) if lines else "  （暂无历史调整记录）"


def _build_prompt(window_summary, history, current_low_max, current_high_min,
                  orig_low_max=None, orig_high_min=None):
    """Build the LLM prompt for threshold adjustment analysis.

    Parameters
    ----------
    window_summary : dict      from WindowStats.summary()
    history : list             previous adjustment records
    current_low_max, current_high_min : float
    orig_low_max, orig_high_min : float or None
        训练开始时的原设定, 用于说明 ±1 m/s 调整带; None 时退化为当前值
        (此时提示词里的调整带只是近似说明)。

    Returns
    -------
    prompt : str
    """
    if orig_low_max is None:
        orig_low_max = current_low_max
    if orig_high_min is None:
        orig_high_min = current_high_min

    s = window_summary
    return _PROMPT_TEMPLATE.format(
        current_low_max="{:.1f}".format(current_low_max),
        current_high_min="{:.1f}".format(current_high_min),
        orig_low_max="{:.1f}".format(orig_low_max),
        orig_high_min="{:.1f}".format(orig_high_min),
        adj_low_min="{:.1f}".format(orig_low_max - 1.0),
        adj_low_max="{:.1f}".format(orig_low_max + 1.0),
        adj_high_min="{:.1f}".format(orig_high_min - 1.0),
        adj_high_max="{:.1f}".format(orig_high_min + 1.0),
        total_steps=s['total_steps'],
        blocks=s['blocks'],
        r1_avg=s['r1_avg'],
        profit_avg=s['profit_avg'],
        violation_rate=s['violation_rate'],
        infeasible_count=s['infeasible_count'],
        gap_pen_total=s['gap_pen_total'],
        gap_pen_per_block=s['gap_pen_per_block'],
        gap_pen_block_rate=s['gap_pen_block_rate'],
        soc_track_err_avg=s['soc_track_err_avg'],
        high_steps=s['high_wind']['steps'],
        high_pct=s['high_wind']['pct'],
        high_r1_avg=s['high_wind']['r1_avg'],
        high_profit_avg=s['high_wind']['profit_avg'],
        high_violation_rate=s['high_wind']['violation_rate'],
        med_steps=s['med_wind']['steps'],
        med_pct=s['med_wind']['pct'],
        med_r1_avg=s['med_wind']['r1_avg'],
        med_profit_avg=s['med_wind']['profit_avg'],
        low_steps=s['low_wind']['steps'],
        low_pct=s['low_wind']['pct'],
        low_r1_avg=s['low_wind']['r1_avg'],
        low_profit_avg=s['low_wind']['profit_avg'],
        history_text=_fmt_history(history),
    )


def _parse_llm_response(response):
    """Parse the LLM JSON reply into (adjust_low_max, adjust_high_min).

    Expected: {"adjust_low_max": <float>, "adjust_high_min": <float>,
               "reasoning": "<text>"}
    Tolerates ```json ... ``` fences. Any parse failure → (0.0, 0.0).
    """
    try:
        text = (response or "").strip()
        if text.startswith("```"):
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1:]
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
            text = text.strip()

        # 容错: 模型偶尔会在 JSON 前后加解释, 取第一个 { ... } 片段
        if not text.startswith("{"):
            i, j = text.find("{"), text.rfind("}")
            if i == -1 or j == -1 or j <= i:
                return 0.0, 0.0
            text = text[i:j + 1]

        data = json.loads(text)
        adjust_low = float(data.get("adjust_low_max", 0.0))
        adjust_high = float(data.get("adjust_high_min", 0.0))
        adjust_low = max(-1.0, min(1.0, adjust_low))
        adjust_high = max(-1.0, min(1.0, adjust_high))
        return adjust_low, adjust_high
    except (json.JSONDecodeError, ValueError, TypeError, KeyError, AttributeError):
        return 0.0, 0.0


def llm_analyze_thresholds(window_summary, history, current_low_max,
                           current_high_min, llm_call_fn=None,
                           orig_low_max=None, orig_high_min=None,
                           return_prompt=False):
    """Ask the LLM for a threshold adjustment.

    Parameters
    ----------
    window_summary : dict          from WindowStats.summary()
    history : list of dict         previous adjustment records
    current_low_max : float
    current_high_min : float
    llm_call_fn : callable or None  prompt_str -> response_str;
                                    None = placeholder mode → (0.0, 0.0)
    orig_low_max, orig_high_min : float or None
        原设定阈值, 写进提示词说明 ±1 m/s 调整带。
    return_prompt : bool
        True 时额外返回提示词 (调试/留档用)。

    Returns
    -------
    (adjust_low_max, adjust_high_min) : (float, float)
        增量 (已夹到 ±1.0); 真正落盘时还要经主循环的 ±1 m/s 调整带钳制 +
        set_wind_thresholds() 的物理钳制。
    若 return_prompt=True, 返回 (adjust_low, adjust_high, prompt)。
    """
    prompt = _build_prompt(window_summary, history, current_low_max,
                           current_high_min, orig_low_max=orig_low_max,
                           orig_high_min=orig_high_min)

    if llm_call_fn is None:
        # 占位模式: 未接 provider / 连通性测试失败 → 不调整 (等价于 base 项目)
        return (0.0, 0.0, prompt) if return_prompt else (0.0, 0.0)

    try:
        response = llm_call_fn(prompt)
        adjust_low, adjust_high = _parse_llm_response(response)
        return (adjust_low, adjust_high, prompt) if return_prompt \
            else (adjust_low, adjust_high)
    except Exception:
        # 任何失败 (网络/鉴权/解析) 都不打断训练
        return (0.0, 0.0, prompt) if return_prompt else (0.0, 0.0)


# =============================================================================
# LLM call-function factory (provider config)
# =============================================================================

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "llm_config.json")


def load_llm_config(config_path=None):
    """Load LLM configuration from JSON (default: td3leprrlrb/llm_config.json).

    Expected JSON::

        {"provider": "deepseek", "api_key": "sk-...", "model": "deepseek-chat"}

    api_key falls back to the provider's environment variable
    (ANTHROPIC_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY) when absent.

    Returns
    -------
    dict with keys: provider, api_key, model
    """
    path = config_path or _DEFAULT_CONFIG_PATH
    config = {}
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)

    provider = config.get("provider", "deepseek")
    api_key = config.get("api_key")
    model = config.get("model")

    if not api_key:
        if provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        elif provider == "deepseek":
            api_key = (os.environ.get("DEEPSEEK_API_KEY")
                       or os.environ.get("OPENAI_API_KEY"))
        else:
            api_key = os.environ.get("OPENAI_API_KEY")

    return {"provider": provider, "api_key": api_key, "model": model}


def create_llm_call_fn(provider=None, api_key=None, model=None,
                       config_path=None):
    """Create an ``llm_call_fn`` (prompt_str -> response_str).

    Explicit arguments override the config file; the config file overrides
    environment variables.

    Raises
    ------
    ImportError  provider SDK not installed (caller should fall back to
                 placeholder mode instead of aborting training).
    ValueError   unknown provider.
    """
    cfg = load_llm_config(config_path)
    provider = provider or cfg["provider"]
    api_key = api_key or cfg["api_key"]
    model = model or cfg["model"]

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        model = model or "claude-haiku-4-5-20251001"

        def call_fn(prompt):
            response = client.messages.create(
                model=model, max_tokens=256, temperature=0.3,
                messages=[{"role": "user", "content": prompt}])
            return response.content[0].text

        return call_fn

    if provider in ("openai", "deepseek"):
        from openai import OpenAI
        if provider == "deepseek":
            client = OpenAI(
                api_key=api_key or os.environ.get("DEEPSEEK_API_KEY")
                or os.environ.get("OPENAI_API_KEY"),
                base_url="https://api.deepseek.com")
            model = model or "deepseek-chat"
        else:
            client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
            model = model or "gpt-4o"

        def call_fn(prompt):
            response = client.chat.completions.create(
                model=model, max_tokens=512, temperature=0.3,
                messages=[{"role": "user", "content": prompt}])
            return response.choices[0].message.content

        return call_fn

    raise ValueError(
        "Unknown provider: {!r}. Use 'anthropic', 'openai' or 'deepseek'."
        .format(provider))
