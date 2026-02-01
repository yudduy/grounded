"""Prompt templates for the discovery loop."""

import numpy as np

# 100 rounds x 5 pts/round = 500 total; show most recent 200 in prompts
MAX_PROMPT_OBSERVATIONS = 200


CHOOSE_SYSTEM = """You are a scientist designing experiments to discover a hidden physical law.
You have {n_inputs} input variables: {input_names}.
Each input has a valid range: {input_ranges}.
You need to choose {n_points} input points that will be most informative
for understanding the relationship y = f({input_args}).

{playbook_section}

Consider:
- Exploring the boundaries of the input space
- Testing one variable at a time while holding others constant
- Probing suspected nonlinearities
- Filling gaps in existing coverage"""

CHOOSE_USER = """Round {round_num}/{total_rounds}.

Previous observations ({n_obs} total):
{observation_summary}

{reflection_section}

Choose {n_points} input points as a Python list of lists.
Format: [[x1_val, x2_val, ...], ...]
IMPORTANT: Your final answer MUST be on the LAST line as ONLY the Python list.
No explanation after it — just the list."""


HYPOTHESIZE_SYSTEM = """You are a scientist analyzing experimental data to discover a hidden physical law.
Variables: {input_names} → y
Valid ranges: {input_ranges}

{playbook_section}

Your task: propose a mathematical equation y = f({input_args}) that best fits
all observations. Use Python/numpy syntax (np.sin, np.cos, np.sqrt, np.exp, np.log, etc.).
IMPORTANT: Use NUMERIC coefficients only (e.g., -4.9 * time**2, not A * time**2).
Do NOT use symbolic parameters like A, B, C — estimate actual numbers from the data."""

HYPOTHESIZE_USER = """Round {round_num}/{total_rounds}.

All observations ({n_obs} points):
{data_table}

{previous_hypotheses}

{reflection_section}

Propose your best equation for y = f({input_args}).
IMPORTANT: Your final answer MUST be on the LAST line, as a plain Python/numpy expression.
No y= prefix, no explanation — just the expression on the last line.
Example final line: 2.5 * time**2 - 0.3 * np.sin(y_offset)"""


REFLECT_SYSTEM = """You are a scientist reflecting on your equation discovery process.
You proposed an equation and evaluated it. Analyze what worked and what didn't.

Consider:
- Did the functional form capture the overall trend?
- Were there systematic residual patterns?
- What aspects of the data are not explained?
- What should be tried differently next round?"""

REFLECT_USER = """Round {round_num}/{total_rounds}.

Proposed equation: {expression}
Train MSE: {train_mse:.6f}
Test MSE: {test_mse:.6f}
Previous best test MSE: {best_mse:.6f}

{residual_summary}

What did you learn? What should change next round?
Be specific and concise (2-3 sentences)."""


def format_observation_summary(inputs, outputs, input_names, max_show=10):
    """Format recent observations as a summary string."""
    n = len(outputs)
    if n == 0:
        return "(no observations yet)"
    header = " | ".join(input_names + ["y"])
    show_n = min(n, max_show)
    rows = []
    for i in range(show_n):
        vals = [f"{inputs[i, j]:.4f}" for j in range(inputs.shape[1])]
        vals.append(f"{outputs[i]:.4f}")
        rows.append(" | ".join(vals))
    summary = header + "\n" + "\n".join(rows)
    if n > max_show:
        summary += f"\n... ({n - max_show} more observations)"
    return summary


def format_data_table(inputs, outputs, input_names, max_show=MAX_PROMPT_OBSERVATIONS):
    """Format all data as a table string."""
    n = len(outputs)
    header = " | ".join(input_names + ["y"])
    rows = []
    show_n = min(n, max_show)
    for i in range(show_n):
        vals = [f"{inputs[i, j]:.4f}" for j in range(inputs.shape[1])]
        vals.append(f"{outputs[i]:.4f}")
        rows.append(" | ".join(vals))
    table = header + "\n" + "\n".join(rows)
    if n > max_show:
        table += f"\n... ({n - max_show} more rows)"
    return table


def format_previous_hypotheses(history, max_show=5):
    """Format recent hypothesis history."""
    if not history:
        return ""
    recent = history[-max_show:]
    lines = ["Previous hypotheses (most recent):"]
    for h in recent:
        lines.append(f"  Round {h['round']}: {h['expression']} (test MSE={h['test_mse']:.6f})")
    return "\n".join(lines)


def format_residual_summary(inputs, targets, predictions, input_names, n_worst=5):
    """Summarize where the model is most wrong."""
    residuals = targets - predictions
    abs_res = np.abs(residuals)
    worst_idx = np.argsort(-abs_res)[:n_worst]
    lines = ["Largest residuals:"]
    for idx in worst_idx:
        inp_str = ", ".join(f"{input_names[j]}={inputs[idx, j]:.3f}"
                            for j in range(inputs.shape[1]))
        lines.append(f"  ({inp_str}) predicted={predictions[idx]:.4f} "
                      f"actual={targets[idx]:.4f} error={residuals[idx]:.4f}")
    return "\n".join(lines)
