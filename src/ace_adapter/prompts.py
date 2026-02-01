"""Equation discovery prompts for ACE Generator/Reflector/Curator.

Customizes the standard ACE prompts for the equation discovery domain.
"""

EQUATION_PLAYBOOK_TEMPLATE = """## EQUATION FORMS
(Functional forms that have shown promise)

## SUCCESSFUL EXAMPLES
(Top equations discovered so far with their MSE)

## PARAMETER STRATEGIES
(Approaches for estimating parameter values)

## EXPLORATION HEURISTICS
(Strategies for choosing informative input points)

## COMMON MISTAKES
(Pitfalls to avoid in equation discovery)
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


def format_bullets_used(state_history, n_recent=5):
    """Format recent playbook bullets used for the reflector."""
    recent = state_history[-n_recent:] if state_history else []
    lines = []
    for h in recent:
        expr = h.get("expression", "?")
        mse = h.get("test_mse", float("inf"))
        lines.append(f"- {expr} (MSE={mse:.6f})")
    return "\n".join(lines) if lines else "(no history)"
