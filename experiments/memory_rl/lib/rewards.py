"""Reward functions for GRPO training in Memory x RL experiment.

Three reward strategies:
- majority_vote_reward: Standard TTRL majority vote (Conditions B, C)
- ace_reward_fn: Majority vote + ACE side-effects for playbook tagging (Condition D)
- novelty_augmented_reward: Majority vote + semantic novelty bonus (Condition E)
"""

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


def majority_vote(answers: List[str]) -> Tuple[str, float]:
    """Compute majority vote winner and confidence.

    Returns: (winner_answer, confidence)
    """
    counter: Counter = Counter()
    for a in answers:
        try:
            normalized = str(int(float(a.replace(",", ""))))
        except (ValueError, TypeError):
            normalized = a.strip()
        counter[normalized] += 1
    if not counter:
        return "", 0.0
    winner, count = counter.most_common(1)[0]
    return winner, count / len(answers)


def _normalize_answer(a: str) -> str:
    """Normalize an answer string for comparison."""
    try:
        return str(int(float(a.replace(",", ""))))
    except (ValueError, TypeError):
        return a.strip()


# ---------------------------------------------------------------------------
# Condition B / C: Pure majority vote reward
# ---------------------------------------------------------------------------

def majority_vote_reward(prompts, completions, **kwargs) -> list[float]:
    """GRPO reward: binary 1.0 if answer matches majority, else 0.0.

    Used by Conditions B (GRPO-only) and C (static playbook).
    """
    from .answer_parsing import parse_answer

    answers = [parse_answer(c) for c in completions]
    winner, _confidence = majority_vote(answers)
    return [1.0 if _normalize_answer(a) == winner else 0.0 for a in answers]


# ---------------------------------------------------------------------------
# Condition D: ACE reward with playbook side-effects
# ---------------------------------------------------------------------------

def ace_reward_fn(prompts, completions, *, state: Dict[str, Any], **kwargs) -> list[float]:
    """GRPO reward with ACE playbook side-effects.

    Same base reward as majority_vote, but also:
    1. Extracts bullet IDs referenced in completions
    2. Tags bullets as helpful/harmful based on correctness
    3. Appends to state["pending_curate"] for deferred curation

    Args:
        prompts: List of prompt strings (or single string)
        completions: List of completion strings
        state: Mutable dict with keys:
            - playbook_mgr: ActivePlaybook instance
            - problem_lookup: {prompt_str: problem_dict}
            - pending_curate: list to append curate items to
            - episode_stats: list for tracking
    """
    from .answer_parsing import parse_answer, check_answer

    answers = [parse_answer(c) for c in completions]
    winner, confidence = majority_vote(answers)

    # Base reward: majority vote
    rewards = [1.0 if _normalize_answer(a) == winner else 0.0 for a in answers]

    # Side-effects: tag bullets, collect curate items
    playbook_mgr = state.get("playbook_mgr")
    if playbook_mgr is None:
        return rewards

    # Look up the ground truth for this prompt.
    # GRPOTrainer passes full chat-templated prompts, but problem_lookup may be
    # keyed on raw problem text. Try exact match first, then substring fallback.
    prompt_key = prompts[0] if isinstance(prompts, list) else str(prompts)
    lookup = state.get("problem_lookup", {})
    problem_dict = lookup.get(prompt_key)
    if problem_dict is None:
        for key, pdict in lookup.items():
            if key in prompt_key:
                problem_dict = pdict
                break
    ground_truth = problem_dict["answer"] if problem_dict else None
    is_correct = check_answer(winner, ground_truth) if ground_truth else False

    # Extract bullet references from completions
    candidates = []
    for c_text, a in zip(completions, answers):
        bullets_used = re.findall(r"\[((?:str|err|sol|gen)-\d{5})\]", c_text)
        candidates.append({
            "answer": a,
            "raw": c_text,
            "bullets_used": list(set(bullets_used)),
        })

    # Tag bullets based on overall correctness
    for c in candidates:
        for bid in c.get("bullets_used", []):
            playbook_mgr.playbook.tag(bid, "helpful" if is_correct else "harmful")

    # Collect for deferred curation (between epochs)
    state.setdefault("pending_curate", []).append({
        "problem": problem_dict["problem"] if problem_dict else "(unknown)",
        "solution": winner,
        "is_correct": is_correct,
        "candidates": candidates,
    })

    state.setdefault("episode_stats", []).append({
        "confidence": confidence,
        "is_correct": is_correct,
        "pb_size": playbook_mgr.playbook.size if hasattr(playbook_mgr, "playbook") else 0,
    })

    return rewards


# ---------------------------------------------------------------------------
# Condition E: Novelty-augmented reward (EVOL-RL inspired)
# ---------------------------------------------------------------------------

def _compute_tfidf_novelty(texts: List[str]) -> List[float]:
    """Compute pairwise dissimilarity among texts using TF-IDF cosine distance.

    Returns a novelty score in [0, 1] for each text: the average cosine distance
    to all other texts. Higher = more novel.

    Uses simple word-level TF-IDF to avoid heavy dependencies in the reward
    hot path (no sentence-transformers).
    """
    if len(texts) <= 1:
        return [0.0] * len(texts)

    # Build vocabulary
    from collections import Counter
    import math

    docs = [Counter(t.lower().split()) for t in texts]
    vocab = set()
    for d in docs:
        vocab.update(d.keys())
    vocab = sorted(vocab)
    vocab_idx = {w: i for i, w in enumerate(vocab)}

    # TF-IDF vectors
    n_docs = len(docs)
    # Document frequency
    df = Counter()
    for d in docs:
        for w in d:
            df[w] += 1

    vectors = []
    for d in docs:
        total_words = sum(d.values()) or 1
        vec = [0.0] * len(vocab)
        for w, count in d.items():
            tf = count / total_words
            idf = math.log((n_docs + 1) / (df[w] + 1)) + 1
            vec[vocab_idx[w]] = tf * idf
        vectors.append(vec)

    # Cosine distances
    def cosine_dist(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a)) or 1e-9
        norm_b = math.sqrt(sum(x * x for x in b)) or 1e-9
        sim = dot / (norm_a * norm_b)
        return 1.0 - max(0.0, min(1.0, sim))

    novelty_scores = []
    for i in range(n_docs):
        dists = [cosine_dist(vectors[i], vectors[j]) for j in range(n_docs) if j != i]
        novelty_scores.append(sum(dists) / len(dists) if dists else 0.0)

    return novelty_scores


def novelty_augmented_reward(
    prompts, completions, *, novelty_weight: float = 0.3, **kwargs
) -> list[float]:
    """GRPO reward with novelty bonus for diverse correct solutions.

    Combined: reward = base_reward + novelty_weight * novelty_score
    Novelty bonus only applied to correct (majority-vote-agreeing) answers.

    Inspired by EVOL-RL: encourages the model to find diverse solution paths
    rather than collapsing to a single template.
    """
    from .answer_parsing import parse_answer

    answers = [parse_answer(c) for c in completions]
    winner, _confidence = majority_vote(answers)

    # Identify correct completions
    correct_mask = [_normalize_answer(a) == winner for a in answers]
    correct_indices = [i for i, m in enumerate(correct_mask) if m]
    correct_texts = [completions[i] for i in correct_indices]

    # Compute novelty among correct completions
    if len(correct_texts) >= 2:
        novelty_scores_correct = _compute_tfidf_novelty(correct_texts)
    else:
        novelty_scores_correct = [0.0] * len(correct_texts)

    # Build reward vector
    rewards = []
    correct_idx_counter = 0
    for i, is_correct in enumerate(correct_mask):
        if is_correct:
            base = 1.0
            novelty = novelty_scores_correct[correct_idx_counter]
            correct_idx_counter += 1
            rewards.append(base + novelty_weight * novelty)
        else:
            rewards.append(0.0)

    return rewards
