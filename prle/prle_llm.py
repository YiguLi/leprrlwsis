"""
PRLE LLM Analysis Module
=========================
Two components:
  1. WindowStats — accumulates per-step metrics over a 1440-step window
  2. llm_analyze_thresholds — calls LLM to suggest threshold adjustments

The LLM call is a placeholder that returns (0.0, 0.0). Prompt engineering
will be addressed in a later phase.
"""

import json
import os


class WindowStats:
    """
    Accumulates per-step training metrics over a 1440-step window.

    Tracks reward totals, violation counts, and per-wind-level statistics
    to provide the LLM with a comprehensive view of the agent's performance
    under the current wind thresholds.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all accumulators for a new window."""
        self.r1_total = 0.0
        self.violation_count = 0        # steps where P_F > 3.0
        self.infeasible_count = 0        # RBB infeasible events
        self.high_wind_steps = 0         # steps classified as HIGH wind
        self.high_wind_r1 = 0.0          # r1 sum during HIGH wind
        self.high_wind_violations = 0    # violations during HIGH wind
        self.med_wind_steps = 0
        self.med_wind_r1 = 0.0
        self.low_wind_steps = 0
        self.low_wind_r1 = 0.0
        self.step_count = 0

    def add(self, r1, wind_level, P_F, is_infeasible):
        """
        Add one step's metrics to the window.

        Parameters
        ----------
        r1 : float
            Upper-level reward for this step.
        wind_level : str
            'LOW', 'MEDIUM', or 'HIGH'.
        P_F : float
            Grid power fluctuation |P_G(t) - P_G(t-1)|.
        is_infeasible : bool
            Whether the RBB step was infeasible (pbrulemin > pbrulemax).
        """
        self.r1_total += r1
        self.step_count += 1

        if wind_level == 'HIGH':
            self.high_wind_steps += 1
            self.high_wind_r1 += r1
            if P_F > 3.0 + 1e-4:
                self.high_wind_violations += 1
        elif wind_level == 'MEDIUM':
            self.med_wind_steps += 1
            self.med_wind_r1 += r1
        else:
            self.low_wind_steps += 1
            self.low_wind_r1 += r1

        if P_F > 3.0 + 1e-4:
            self.violation_count += 1
        if is_infeasible:
            self.infeasible_count += 1

    def summary(self):
        """
        Return a dict of summary statistics for LLM consumption.

        Returns
        -------
        dict with keys:
            total_steps, r1_total, r1_avg, violation_rate, infeasible_count,
            high_wind, med_wind, low_wind (each with steps, r1_avg,
            and violation_rate for high_wind only).
        """
        n = max(self.step_count, 1)
        return {
            'total_steps': self.step_count,
            'r1_total': round(self.r1_total, 2),
            'r1_avg': round(self.r1_total / n, 4),
            'violation_rate': round(self.violation_count / n, 4),
            'infeasible_count': self.infeasible_count,
            'high_wind': {
                'steps': self.high_wind_steps,
                'pct': round(100.0 * self.high_wind_steps / n, 1),
                'r1_avg': round(self.high_wind_r1 / max(self.high_wind_steps, 1), 4),
                'violation_rate': round(
                    self.high_wind_violations / max(self.high_wind_steps, 1), 4),
            },
            'med_wind': {
                'steps': self.med_wind_steps,
                'pct': round(100.0 * self.med_wind_steps / n, 1),
                'r1_avg': round(self.med_wind_r1 / max(self.med_wind_steps, 1), 4),
            },
            'low_wind': {
                'steps': self.low_wind_steps,
                'pct': round(100.0 * self.low_wind_steps / n, 1),
                'r1_avg': round(self.low_wind_r1 / max(self.low_wind_steps, 1), 4),
            },
        }


# =============================================================================
# Prompt Template (Chinese production version with JSON output)
# =============================================================================

_PROMPT_TEMPLATE = """你是一个风储混合发电系统的风险调度专家。系统包含5台风机和1组锂电池储能（BESS），
通过强化学习智能体（SP-DDPG）在电力现货市场中进行实时竞价与功率调度。

电网要求相邻时刻的并网功率波动不超过3.0 MW，否则记为一次越限违规。

═══════════════════════════════════════════
当前风险分段阈值
═══════════════════════════════════════════
低风速（LOW）：  u < {current_low_max} m/s
中风速（MEDIUM）：{current_low_max} ≤ u < {current_high_min} m/s
高风速（HIGH）：  u ≥ {current_high_min} m/s

═══════════════════════════════════════════
分段奖励系数表（按风速×电价共9档）
═══════════════════════════════════════════
风速↓ / 电价→   LOW(300)   MEDIUM(580)   HIGH(980)
HIGH            WT:0.7~0.9, BESS:1.0   ← 抑制激进策略，降低盈利但控风险
MEDIUM          WT:1.0~1.2, BESS:1.0   ← 正常激励
LOW             WT:1.0~1.2, BESS:1.0   ← 鼓励多发
（BESS 收益权重 wB ≡ 1.0，不随电价档/风速档/SOC 变化）

风速档还决定【小额波动系数】c1_small（波动成本中 P_F 的线性项）：
HIGH=0.5 / MEDIUM=0.2 / LOW=0.1（越限项 10·(P_F−3) 不随档变化）。

核心逻辑：样本被归入HIGH风速 → turbine奖励系数被压低、小额波动系数抬高(0.2→0.5)
→ 该步盈利贡献被抑制、且波动被更重地惩罚。
因此：
  - 调高HIGH_MIN → 更少样本归入HIGH → 整体盈利空间增大，但风险敞口也增大
  - 调低HIGH_MIN → 更多样本归入HIGH → 更保守，盈利空间缩小，但更安全
  - 调高LOW_MAX → 更多样本归入LOW（WT×1.2、小额波动系数 0.2→0.1）→ 中等风速也被鼓励多发
  - 调低LOW_MAX → 更少样本归入LOW → 中等风速趋于保守

═══════════════════════════════════════════
当前窗口统计数据（1440步 = 1天）
═══════════════════════════════════════════
总步数：{total_steps}
平均r1（盈利代理）：{r1_avg}
整体越限率：{violation_rate}（越限 = |ΔP_G| > 3.0 MW）
不可行事件数：{infeasible_count}（物理约束与波动约束无交集，违规物理上无法避免）

按风速分段：
  HIGH   风速：{high_steps}步（{high_pct}%），
         平均r1 = {high_r1_avg}，越限率 = {high_violation_rate}
  MEDIUM 风速：{med_steps}步（{med_pct}%），
         平均r1 = {med_r1_avg}
  LOW    风速：{low_steps}步（{low_pct}%），
         平均r1 = {low_r1_avg}

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
  → 是：下调HIGH_MIN（扩大HIGH区），即使牺牲盈利也要控风险。建议幅度-0.3~-0.5。
  → 否：进入第2步。

2.【盈利优化】整体越限率是否很低（<1%）且HIGH越限率也低（<2%）？
  → 是：系统当前过于保守，可以上调HIGH_MIN（缩小HIGH区）释放盈利空间。建议幅度+0.1~+0.3。
  → 否：进入第3步。

3.【不可行事件】infeasible_count是否持续增长？
  → 是：风速已超出系统的物理调节能力，不应继续上调HIGH_MIN。
  → 同时考虑下调HIGH_MIN以提前抑制高风险动作。

4.【LOW/MEDIUM边界微调】LOW和MEDIUM的奖励系数差异主要在BESS端。
  除非LOW风速占比过大导致过度激励，否则LOW_MAX通常无需大幅调整。
  建议将LOW_MAX的调整幅度控制在±0.1~±0.2。

5.【振荡抑制】检查历史记录：如果上一次调整方向与本次相反，说明系统在震荡。
  → 将本次调整幅度减半（阻尼），或维持不变。

6.【幅度约束】单次调整幅度不得超过±0.5 m/s。优先采用小幅渐进策略（±0.1~±0.2），
  仅在越限率显著超标（>8%）时使用较大幅度（±0.3~±0.5）。

═══════════════════════════════════════════
输出格式（严格遵守）
═══════════════════════════════════════════
只输出一行合法的JSON，不要包含```json```代码块标记，不要包含任何其他文字。
JSON格式如下：
{{"adjust_low_max": <float>, "adjust_high_min": <float>, "reasoning": "<一句话理由>"}}

示例输出：
{{"adjust_low_max": 0.0, "adjust_high_min": -0.3, "reasoning": "HIGH风速越限率达7.2%，超过5%安全线，下调HIGH_MIN以扩大高风速保守区，降低违规风险"}}"""


def _build_prompt(window_summary, history, current_low_max, current_high_min):
    """
    Build the LLM prompt for threshold adjustment analysis.

    Parameters
    ----------
    window_summary : dict
        From WindowStats.summary().
    history : list
        Previous threshold adjustment records (last 5).
    current_low_max : float
    current_high_min : float

    Returns
    -------
    prompt : str
    """
    # Build history text
    history_lines = []
    for i, h in enumerate(history[-5:]):
        history_lines.append(
            f"  窗口{i+1}: Step {h.get('step', '?')}, "
            f"LOW_MAX {h['old_low_max']:.1f}→{h['new_low_max']:.1f}, "
            f"HIGH_MIN {h['old_high_min']:.1f}→{h['new_high_min']:.1f}"
        )
    history_text = "\n".join(history_lines) if history_lines else "  （暂无历史调整记录）"

    prompt = _PROMPT_TEMPLATE.format(
        current_low_max=f"{current_low_max:.1f}",
        current_high_min=f"{current_high_min:.1f}",
        total_steps=window_summary['total_steps'],
        r1_avg=window_summary['r1_avg'],
        violation_rate=window_summary['violation_rate'],
        infeasible_count=window_summary['infeasible_count'],
        high_steps=window_summary['high_wind']['steps'],
        high_pct=window_summary['high_wind']['pct'],
        high_r1_avg=window_summary['high_wind']['r1_avg'],
        high_violation_rate=window_summary['high_wind']['violation_rate'],
        med_steps=window_summary['med_wind']['steps'],
        med_pct=window_summary['med_wind']['pct'],
        med_r1_avg=window_summary['med_wind']['r1_avg'],
        low_steps=window_summary['low_wind']['steps'],
        low_pct=window_summary['low_wind']['pct'],
        low_r1_avg=window_summary['low_wind']['r1_avg'],
        history_text=history_text,
    )

    return prompt


def llm_analyze_thresholds(window_summary, history, current_low_max, current_high_min,
                           llm_call_fn=None):
    """
    Analyze window statistics and suggest threshold adjustments.

    Parameters
    ----------
    window_summary : dict
        From WindowStats.summary().
    history : list of dicts
        Previous threshold adjustment records.
    current_low_max : float
        Current WIND_LOW_MAX threshold.
    current_high_min : float
        Current WIND_HIGH_MIN threshold.
    llm_call_fn : callable or None
        Function that takes a prompt string and returns the LLM response string.
        If None (default), returns (0.0, 0.0) — placeholder mode.

    Returns
    -------
    (adjust_low_max, adjust_high_min) : (float, float)
        Adjustment amounts in [-1.0, 1.0]. These will be ADDED to current thresholds
        and then validated/clamped by set_wind_thresholds().
    """
    prompt = _build_prompt(window_summary, history, current_low_max, current_high_min)

    if llm_call_fn is None:
        # Placeholder: no LLM connected yet, return zero adjustment
        return 0.0, 0.0

    try:
        response = llm_call_fn(prompt)
        adjust_low, adjust_high = _parse_llm_response(response)
        return adjust_low, adjust_high
    except Exception:
        # On any failure (API error, parse error, etc.), return no adjustment
        return 0.0, 0.0


def _parse_llm_response(response):
    """
    Parse LLM JSON response into adjustment values.

    Expected format:
        {"adjust_low_max": <float>, "adjust_high_min": <float>, "reasoning": "<text>"}

    Handles LLMs that may wrap output in ```json code blocks.

    Parameters
    ----------
    response : str

    Returns
    -------
    (adjust_low_max, adjust_high_min) : (float, float)
    """
    try:
        text = response.strip()

        # Strip ```json ... ``` code block wrapper if present
        if text.startswith("```"):
            # Remove opening ```json or ```
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        data = json.loads(text)

        adjust_low = float(data.get("adjust_low_max", 0.0))
        adjust_high = float(data.get("adjust_high_min", 0.0))

        # Clamp to valid range
        adjust_low = max(-1.0, min(1.0, adjust_low))
        adjust_high = max(-1.0, min(1.0, adjust_high))

        return adjust_low, adjust_high

    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
        # On any parse failure, return no adjustment
        return 0.0, 0.0


# =============================================================================
# LLM Call Function Factory
# =============================================================================

# =============================================================================
# Default config path
# =============================================================================

_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "llm_config.json")


def load_llm_config(config_path=None):
    """
    Load LLM configuration from a JSON file.

    Looks for config at `config_path` if given, otherwise at
    `prle/llm_config.json`. Falls back to environment variables
    for the API key if not present in config.

    Expected JSON format:
    {
        "provider": "anthropic",
        "api_key": "sk-ant-...",      // optional: falls back to env var
        "model": "claude-haiku-4-5-20251001"  // optional: uses provider default
    }

    Parameters
    ----------
    config_path : str or None
        Path to config JSON file.

    Returns
    -------
    dict with keys: provider, api_key, model
    """
    import os

    path = config_path or _DEFAULT_CONFIG_PATH
    config = {}

    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        # No config file found — fall back to env vars and defaults
        pass

    provider = config.get("provider", "anthropic")
    api_key = config.get("api_key")  # None if not in config
    model = config.get("model")      # None → use provider default

    # If api_key not in config, try environment variable
    if not api_key:
        if provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        else:
            api_key = os.environ.get("OPENAI_API_KEY")

    return {"provider": provider, "api_key": api_key, "model": model}


def create_llm_call_fn(provider=None, api_key=None, model=None, config_path=None):
    """
    Create an llm_call_fn suitable for passing to llm_analyze_thresholds.

    Configuration is loaded from a JSON config file (default: prle/llm_config.json).
    Explicitly passed arguments override the config file.
    Environment variables are used as fallback for api_key.

    Parameters
    ----------
    provider : str or None
        "anthropic" | "openai" | "deepseek".
        If None, read from config file.
    api_key : str or None
        API key. If None, read from config file, then env var.
    model : str or None
        Model ID. If None, uses provider default.
    config_path : str or None
        Path to config JSON file. If None, uses prle/llm_config.json.

    Returns
    -------
    callable : prompt_str → response_str
    """
    import os

    # Load from config file; passed args take precedence
    cfg = load_llm_config(config_path)
    provider = provider or cfg["provider"]
    api_key = api_key or cfg["api_key"]
    model = model or cfg["model"]

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        model = model or "claude-haiku-4-5-20251001"

        def call_fn(prompt):
            response = client.messages.create(
                model=model,
                max_tokens=256,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        return call_fn

    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        model = model or "gpt-4o"

        def call_fn(prompt):
            response = client.chat.completions.create(
                model=model,
                max_tokens=256,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content

        return call_fn

    elif provider == "deepseek":
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        model = model or "deepseek-chat"

        def call_fn(prompt):
            response = client.chat.completions.create(
                model=model,
                max_tokens=256,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content

        return call_fn

    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'anthropic', 'openai', or 'deepseek'.")
