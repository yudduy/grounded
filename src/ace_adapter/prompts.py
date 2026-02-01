"""Equation discovery prompts for ACE Generator/Reflector/Curator.

Customizes the standard ACE prompts for the equation discovery domain.
"""

EQUATION_PLAYBOOK_TEMPLATE = """## EQUATION FORMS
[eq-00001] helpful=0 harmful=0 :: Try simple polynomial forms before complex nonlinear ones

## SUCCESSFUL EXAMPLES
[eq-00002] helpful=0 harmful=0 :: Track top equations with their MSE for reference

## PARAMETER STRATEGIES
[eq-00003] helpful=0 harmful=0 :: Use least-squares fitting after choosing functional form

## EXPLORATION HEURISTICS
[eq-00004] helpful=0 harmful=0 :: Vary one input dimension at a time to isolate dependencies

## COMMON MISTAKES
[eq-00005] helpful=0 harmful=0 :: Avoid overfitting by preferring simpler expressions
"""

EQUATION_GENERATOR_CONTEXT = """You are discovering a hidden physical law y = f({input_args}).
You have {n_obs} observations. Use the playbook strategies to guide your hypothesis.
Focus on the functional form first, then worry about exact parameters.

Data summary:
{data_summary}
"""

EQUATION_REFLECTOR_CONTEXT = """Evaluate how well the proposed equation explains the data.
Tag each playbook bullet you used as 'helpful' or 'harmful' based on
whether following that strategy improved or worsened your prediction.

Proposed: {expression}
Train MSE: {train_mse:.6f}
Test MSE: {test_mse:.6f}
Previous best: {best_mse:.6f}
Improvement: {improved}
"""

EQUATION_CURATOR_CONTEXT = """Update the equation discovery playbook based on recent reflections.
Add new insights, remove unhelpful strategies, merge similar ones.
Keep the playbook focused and actionable.

Sections: EQUATION FORMS, PARAMETER STRATEGIES, EXPLORATION HEURISTICS, COMMON MISTAKES

Recent performance: {performance_summary}
"""


def format_bullets_used(state_history, n_recent=5, playbook=None):
    """Format recent playbook bullets used for the reflector.

    Parses the current playbook for canonical bullet lines ([id] ... :: content)
    and returns them in the format the ACE Reflector expects for tagging.
    Falls back to expression history if no playbook bullets are available.
    """
    import re

    bullets = []
    if playbook:
        for line in playbook.splitlines():
            m = re.match(r"^\[([a-z]+-\d+)\]\s+(helpful=\d+\s+harmful=\d+)\s+::\s+(.+)$", line.strip())
            if m:
                bullet_id, counts, content = m.group(1), m.group(2), m.group(3)
                bullets.append(f"[{bullet_id}] {counts} :: {content}")

    if bullets:
        return "\n".join(bullets)

    # Fallback: summarize recent history when no playbook is available
    recent = state_history[-n_recent:] if state_history else []
    lines = []
    for h in recent:
        expr = h.get("expression", "?")
        mse = h.get("test_mse", float("inf"))
        lines.append(f"- {expr} (MSE={mse:.6f})")
    return "\n".join(lines) if lines else "(no history)"
