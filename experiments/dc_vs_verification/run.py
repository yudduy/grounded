#!/usr/bin/env python3
"""
DC vs Verification on AIME 2024
================================
4 conditions x 3 seeds on 30 AIME 2024 problems using Qwen2.5-7B-Instruct via vLLM.

Conditions:
  1. Baseline      — CoT Pass@1, no memory
  2. Self-Consistency — N=16 majority vote
  3. Dynamic Cheatsheet (DC) — N=8 candidates, reflect+curate loop, self-consistency answer selection
  4. DC + Outcome Verification — same as #3 but only verified-correct strategies enter playbook
"""

import argparse
import asyncio
import copy
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import AsyncOpenAI, OpenAI

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
AIME_CSV = REPO_ROOT / "ShinkaEvolve" / "examples" / "adas_aime" / "AIME_Dataset_1983_2025.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ---------------------------------------------------------------------------
# Answer parsing (adapted from ShinkaEvolve/examples/adas_aime/utils.py)
# We inline these functions because utils.py has a top-level `import backoff`
# that may not be installed, and we only need a few functions.
# ---------------------------------------------------------------------------
ANSWER_REGEX = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def extract_numeric_answer(text: str) -> str:
    matches = ANSWER_REGEX.findall(text.replace(",", ""))
    if not matches:
        return text.strip()
    result = matches[-1].lstrip("0")
    return result if result else "0"


def last_boxed_only_string(string: str) -> str:
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
    if idx < 0:
        return ""
    brace_idx = string.find("{", idx)
    if brace_idx < 0:
        return ""
    level = 0
    for i in range(brace_idx, len(string)):
        if string[i] == "{":
            level += 1
        elif string[i] == "}":
            level -= 1
            if level == 0:
                return string[idx : i + 1]
    return ""


def clean_answer(s):
    s = s.replace("\\dfrac", "\\frac")
    s = s.replace("x \\in", "")
    s = re.sub(r"\\mathbf\s*{([^}]*)}", r"\1", s)
    s = re.sub(r"\\textbf\s*{([^}]*)}", r"\1", s)
    return s


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]
    left = "\\boxed{"
    if not s.startswith(left):
        return None
    assert s[-1] == "}"
    return clean_answer(s[len(left) : -1])


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a, b = substr[0], substr[1]
                if b != "{":
                    new_str += "{" + a + "}{" + b + "}" + substr[2:]
                else:
                    new_str += "{" + a + "}" + b + substr[2:]
    return new_str


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a, b = int(a), int(b)
        assert string == "{}/{}".format(a, b)
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except (AssertionError, ValueError):
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    return string


def strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]
    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    if string == "5.5":
        string = "\\frac{11}{2}"
    string = fix_a_slash_b(string)
    return string


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def parse_answer(raw: str) -> str:
    """Extract answer from model output. Try \\boxed first, then #### pattern, then last number."""
    boxed = last_boxed_only_string(raw)
    if boxed:
        inner = remove_boxed(boxed)
        if inner is not None:
            return inner.strip()
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", raw)
    if m:
        return m.group(1).replace(",", "").strip()
    return extract_numeric_answer(raw)


def check_answer(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth."""
    if is_equiv(predicted, ground_truth):
        return True
    # AIME answers are integers 000-999; try numeric comparison
    try:
        p = int(float(predicted.replace(",", "")))
        g = int(float(ground_truth.replace(",", "")))
        return p == g
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_aime_2024() -> List[dict]:
    """Load AIME 2024 problems from the CSV."""
    problems = []
    with open(AIME_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row["Year"]).strip() == "2024":
                problems.append({
                    "id": row["ID"],
                    "problem": row["problem"],
                    "answer": str(int(row["answer"])),  # strip leading zeros
                })
    return problems


# ---------------------------------------------------------------------------
# Playbook (adapted from notebooks/search_augmented_ace_poc.ipynb)
# ---------------------------------------------------------------------------
@dataclass
class Bullet:
    id: str
    section: str
    content: str
    helpful: int = 0
    harmful: int = 0

    def to_str(self) -> str:
        return f"[{self.id}] helpful={self.helpful} harmful={self.harmful} :: {self.content}"


@dataclass
class Playbook:
    bullets: List[Bullet] = field(default_factory=list)
    _next_id: int = 1

    def add(self, section: str, content: str) -> str:
        prefix = {"STRATEGIES": "str", "COMMON_MISTAKES": "err", "SOLUTION_PATTERNS": "sol"}.get(section, "gen")
        bid = f"{prefix}-{self._next_id:05d}"
        self._next_id += 1
        self.bullets.append(Bullet(id=bid, section=section, content=content))
        return bid

    def remove(self, bid: str):
        self.bullets = [b for b in self.bullets if b.id != bid]

    def update(self, bid: str, content: str):
        for b in self.bullets:
            if b.id == bid:
                b.content = content
                return

    def tag(self, bid: str, label: str):
        for b in self.bullets:
            if b.id == bid:
                if label == "helpful":
                    b.helpful += 1
                elif label == "harmful":
                    b.harmful += 1

    def to_str(self) -> str:
        sections = defaultdict(list)
        for b in self.bullets:
            sections[b.section].append(b.to_str())
        parts = []
        for sec in ["STRATEGIES", "COMMON_MISTAKES", "SOLUTION_PATTERNS"]:
            if sections[sec]:
                parts.append(f"## {sec}")
                parts.extend(sections[sec])
        return "\n".join(parts) if parts else "(empty playbook)"

    def copy(self) -> "Playbook":
        return copy.deepcopy(self)

    @property
    def size(self) -> int:
        return len(self.bullets)


MAX_BULLETS = 20


def make_initial_playbook() -> Playbook:
    pb = Playbook()
    pb.add("STRATEGIES", "AIME problems have integer answers from 000 to 999. Always give a non-negative integer.")
    pb.add("STRATEGIES", "Break complex problems into smaller sub-problems and solve each step carefully.")
    pb.add("COMMON_MISTAKES", "Watch for off-by-one errors in counting and combinatorics problems.")
    return pb


# ---------------------------------------------------------------------------
# LLM interface
# ---------------------------------------------------------------------------
VLLM_PORT = 8000
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_CONCURRENT = 64

client: Optional[OpenAI] = None
aclient: Optional[AsyncOpenAI] = None
_semaphore: Optional[asyncio.Semaphore] = None
call_counter: Dict[str, int] = defaultdict(int)


def init_clients():
    global client, aclient, _semaphore
    client = OpenAI(base_url=f"http://localhost:{VLLM_PORT}/v1", api_key="dummy")
    aclient = AsyncOpenAI(base_url=f"http://localhost:{VLLM_PORT}/v1", api_key="dummy")
    _semaphore = asyncio.Semaphore(MAX_CONCURRENT)


async def llm_call_async(system: str, user: str, role: str = "generate",
                         temperature: float = 0.7, max_tokens: int = 2048) -> str:
    call_counter[role] += 1
    async with _semaphore:
        try:
            resp = await aclient.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM call failed ({role}): {e}", file=sys.stderr)
            return ""


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------
def _build_generate_system(playbook: Optional[Playbook] = None) -> str:
    base = (
        "You are an expert math competition solver. Solve the problem step-by-step.\n"
        "Show all your work clearly. At the end, put your final integer answer inside \\boxed{}.\n"
        "AIME answers are always integers from 0 to 999.\n"
    )
    if playbook and playbook.size > 0:
        base += f"\nPLAYBOOK (use these strategies, reference IDs like [str-00001]):\n{playbook.to_str()}"
    return base


async def generate_one(question: str, playbook: Optional[Playbook] = None,
                       temperature: float = 0.7) -> Tuple[str, List[str], str]:
    """Generate a single solution. Returns (answer, bullets_used, raw_response)."""
    system = _build_generate_system(playbook)
    user = f"Solve this AIME problem:\n\n{question}"
    raw = await llm_call_async(system, user, role="generate", temperature=temperature)
    answer = parse_answer(raw)
    # Extract bullet references
    bullets_used = re.findall(r"\[(str|err|sol|gen)-\d{5}\]", raw)
    bullets_used = list(set(bullets_used))
    return answer, bullets_used, raw


async def generate_n(question: str, n: int, playbook: Optional[Playbook] = None,
                     temperature: float = 0.7) -> List[Tuple[str, List[str], str]]:
    """Generate N solutions in parallel."""
    tasks = [generate_one(question, playbook, temperature) for _ in range(n)]
    return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Majority vote
# ---------------------------------------------------------------------------
def majority_vote(answers: List[str]) -> str:
    """Return the most common answer, breaking ties by first occurrence."""
    counter = Counter()
    for a in answers:
        # Normalize: try int conversion for AIME
        try:
            normalized = str(int(float(a.replace(",", ""))))
        except (ValueError, TypeError):
            normalized = a.strip()
        counter[normalized] += 1
    if not counter:
        return ""
    return counter.most_common(1)[0][0]


def majority_vote_confidence(answers: List[str]) -> Tuple[str, float]:
    """Return (winner, confidence) where confidence = fraction of votes for winner."""
    counter = Counter()
    for a in answers:
        try:
            counter[str(int(float(a.replace(",", ""))))] += 1
        except (ValueError, TypeError):
            counter[a.strip()] += 1
    if not counter:
        return "", 0.0
    winner, count = counter.most_common(1)[0]
    return winner, count / len(answers)


# ---------------------------------------------------------------------------
# Reflect
# ---------------------------------------------------------------------------
async def reflect_async(question: str, raw_response: str, predicted: str, ground_truth: str,
                        bullets_used: List[str], playbook: Playbook, is_correct: bool) -> Tuple[str, Dict[str, str]]:
    """Reflect on a solution attempt. Returns (reflection_text, {bullet_id: tag})."""
    feedback = "CORRECT" if is_correct else f"INCORRECT (predicted {predicted}, expected {ground_truth})"
    bullets_text = "\n".join(f"  {b.to_str()}" for b in playbook.bullets if b.id in bullets_used)
    if not bullets_text:
        bullets_text = "  (none referenced)"

    system = (
        "You are a math reasoning analyst. Analyze the solution and whether playbook strategies helped.\n"
        'For each bullet ID used, output a JSON line: {"id": "str-00001", "tag": "helpful"}\n'
        "Tags: helpful, harmful, neutral.\n"
        "End with a reflection paragraph about what mathematical insight was key."
    )
    user = (
        f"Problem: {question}\n\n"
        f"Solution:\n{raw_response}\n\n"
        f"Result: {feedback}\n\n"
        f"Bullets referenced:\n{bullets_text}"
    )
    raw = await llm_call_async(system, user, role="reflect", temperature=0.3)
    tags = {}
    for m in re.finditer(r'"id"\s*:\s*"([^"]+)".*?"tag"\s*:\s*"(helpful|harmful|neutral)"', raw):
        bid, tag = m.group(1), m.group(2)
        if bid in bullets_used:
            tags[bid] = tag
    # Default: correct → helpful, incorrect → harmful
    if not tags and bullets_used:
        default_tag = "helpful" if is_correct else "harmful"
        for bid in bullets_used:
            tags[bid] = default_tag
    return raw, tags


# ---------------------------------------------------------------------------
# Curate
# ---------------------------------------------------------------------------
async def curate_async(playbook: Playbook, reflection: str, question: str) -> Playbook:
    """Curate the playbook based on reflection."""
    pb = playbook.copy()
    pb_text = pb.to_str()
    system = (
        "You are a playbook curator for math competition solving. Based on the reflection, "
        "propose operations to improve the playbook.\n"
        "Output a JSON array of operations:\n"
        '[{"op": "ADD", "section": "STRATEGIES", "content": "new insight"},\n'
        ' {"op": "UPDATE", "id": "str-00001", "content": "refined text"},\n'
        ' {"op": "DELETE", "id": "err-00002"}]\n'
        f"Sections: STRATEGIES, COMMON_MISTAKES, SOLUTION_PATTERNS\n"
        f"Max bullets: {MAX_BULLETS}. Current: {pb.size}.\n"
        "Only propose operations clearly supported by the reflection. Keep it minimal."
    )
    user = (
        f"Question: {question}\n"
        f"Current playbook:\n{pb_text}\n\n"
        f"Reflection:\n{reflection}"
    )
    raw = await llm_call_async(system, user, role="curator", temperature=0.4)

    # Parse JSON operations
    json_match = re.search(r"\[.*\]", raw, re.DOTALL)
    if json_match:
        try:
            ops = json.loads(json_match.group())
        except json.JSONDecodeError:
            ops = []
    else:
        ops = []

    original_next_id = pb._next_id
    for op in ops:
        try:
            if op.get("op") == "ADD" and pb.size < MAX_BULLETS:
                pb.add(op.get("section", "STRATEGIES"), op.get("content", ""))
            elif op.get("op") == "UPDATE" and op.get("id"):
                pb.update(op["id"], op.get("content", ""))
            elif op.get("op") == "DELETE" and op.get("id"):
                pb.remove(op["id"])
        except Exception:
            pass

    if pb.size == 0:
        pb = make_initial_playbook()
        pb._next_id = original_next_id
    return pb


# ---------------------------------------------------------------------------
# Condition runners
# ---------------------------------------------------------------------------

async def run_baseline(problems: List[dict]) -> List[dict]:
    """Condition 1: CoT Pass@1, no memory."""
    results = []
    tasks = []
    for p in problems:
        tasks.append(generate_one(p["problem"], playbook=None, temperature=0.0))

    responses = await asyncio.gather(*tasks)
    for p, (answer, _, raw) in zip(problems, responses):
        correct = check_answer(answer, p["answer"])
        results.append({
            "id": p["id"],
            "predicted": answer,
            "ground_truth": p["answer"],
            "correct": correct,
            "raw": raw,
        })
    return results


async def run_self_consistency(problems: List[dict], n: int = 16) -> List[dict]:
    """Condition 2: N=16 majority vote per problem."""
    results = []
    for p in problems:
        responses = await generate_n(p["problem"], n, playbook=None, temperature=0.7)
        answers = [r[0] for r in responses]
        winner = majority_vote(answers)
        correct = check_answer(winner, p["answer"])
        results.append({
            "id": p["id"],
            "predicted": winner,
            "ground_truth": p["answer"],
            "correct": correct,
            "n_candidates": n,
            "answer_distribution": dict(Counter(answers).most_common()),
        })
    return results


async def run_dc(problems: List[dict], n_candidates: int = 8) -> List[dict]:
    """
    Condition 3: Dynamic Cheatsheet.
    N=8 candidates per problem, majority vote for answer selection.
    Reflect+curate loop evolves playbook across problems.
    """
    playbook = make_initial_playbook()
    results = []

    for i, p in enumerate(problems):
        # Generate N candidates using current playbook
        responses = await generate_n(p["problem"], n_candidates, playbook=playbook, temperature=0.7)
        answers = [r[0] for r in responses]
        winner, confidence = majority_vote_confidence(answers)
        correct = check_answer(winner, p["answer"])

        # Find best response (one that matches majority vote)
        best_raw = ""
        best_bullets = []
        for ans, bullets, raw in responses:
            try:
                if str(int(float(ans.replace(",", "")))) == winner:
                    best_raw = raw
                    best_bullets = bullets
                    break
            except (ValueError, TypeError):
                if ans.strip() == winner:
                    best_raw = raw
                    best_bullets = bullets
                    break
        if not best_raw:
            best_raw = responses[0][2]
            best_bullets = responses[0][1]

        # Reflect using self-consistency confidence as correctness proxy
        # High consensus (>50%) treated as likely correct; low as likely wrong
        is_confident = confidence > 0.5
        reflection, tags = await reflect_async(
            p["problem"], best_raw, winner, "N/A (self-consistency)",
            best_bullets, playbook, is_correct=is_confident
        )
        for bid, tag in tags.items():
            playbook.tag(bid, tag)

        # Curate
        playbook = await curate_async(playbook, reflection, p["problem"])

        results.append({
            "id": p["id"],
            "predicted": winner,
            "ground_truth": p["answer"],
            "correct": correct,
            "problem_idx": i,
            "playbook_size": playbook.size,
            "n_candidates": n_candidates,
            "answer_distribution": dict(Counter(answers).most_common()),
        })
        print(f"  DC [{i+1}/{len(problems)}] {p['id']}: {'Y' if correct else 'N'} (pred={winner}, gt={p['answer']}, pb_size={playbook.size})")

    return results


async def run_dc_verified(problems: List[dict], n_candidates: int = 8) -> List[dict]:
    """
    Condition 4: DC + Outcome Verification.
    Same as DC, but only verified-correct strategies enter the playbook.
    Uses ground truth as a perfect verifier (ceiling estimate).
    """
    playbook = make_initial_playbook()
    results = []

    for i, p in enumerate(problems):
        responses = await generate_n(p["problem"], n_candidates, playbook=playbook, temperature=0.7)
        answers = [r[0] for r in responses]
        winner = majority_vote(answers)
        correct = check_answer(winner, p["answer"])

        # Find a CORRECT response if any exist (use ground truth for verification)
        verified_raw = ""
        verified_bullets = []
        any_correct = False
        for ans, bullets, raw in responses:
            if check_answer(ans, p["answer"]):
                verified_raw = raw
                verified_bullets = bullets
                any_correct = True
                break

        if any_correct:
            # Reflect and curate ONLY on verified-correct solutions
            reflection, tags = await reflect_async(
                p["problem"], verified_raw, p["answer"], p["answer"],
                verified_bullets, playbook, is_correct=True
            )
            for bid, tag in tags.items():
                playbook.tag(bid, tag)
            playbook = await curate_async(playbook, reflection, p["problem"])
        else:
            # No correct solution found — reflect on failure but be cautious with curation
            best_raw = responses[0][2]
            best_bullets = responses[0][1]
            reflection, tags = await reflect_async(
                p["problem"], best_raw, winner, p["answer"],
                best_bullets, playbook, is_correct=False
            )
            for bid, tag in tags.items():
                playbook.tag(bid, tag)
            # Still curate (to learn from mistakes) but tag as unverified
            playbook = await curate_async(playbook, reflection, p["problem"])

        results.append({
            "id": p["id"],
            "predicted": winner,
            "ground_truth": p["answer"],
            "correct": correct,
            "any_candidate_correct": any_correct,
            "problem_idx": i,
            "playbook_size": playbook.size,
            "n_candidates": n_candidates,
            "answer_distribution": dict(Counter(answers).most_common()),
        })
        print(f"  DC+V [{i+1}/{len(problems)}] {p['id']}: {'Y' if correct else 'N'} (verified={any_correct}, pb_size={playbook.size})")

    return results


# ---------------------------------------------------------------------------
# vLLM server management
# ---------------------------------------------------------------------------
def start_vllm() -> subprocess.Popen:
    """Start vLLM server and wait until ready."""
    print(f"Starting vLLM server with {MODEL_NAME}...")
    vllm_log = RESULTS_DIR / "vllm_server.log"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_fh = open(vllm_log, "w")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL_NAME,
            "--port", str(VLLM_PORT),
            "--dtype", "bfloat16",
            "--gpu-memory-utilization", "0.90",
            "--max-model-len", "8192",
            "--max-num-seqs", "512",
            "--max-num-batched-tokens", "16384",
            "--enable-prefix-caching",
        ],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    print(f"vLLM PID: {proc.pid}, log: {vllm_log}")
    # Wait for server (up to 10 minutes)
    init_clients()
    for attempt in range(300):
        if proc.poll() is not None:
            log_fh.close()
            with open(vllm_log) as f:
                tail = f.read()[-2000:]
            raise RuntimeError(f"vLLM process died (exit={proc.returncode}). Last log:\n{tail}")
        try:
            client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1,
            )
            print(f"vLLM server ready after ~{attempt*2}s (PID: {proc.pid})")
            return proc
        except Exception:
            time.sleep(2)
    log_fh.close()
    with open(vllm_log) as f:
        tail = f.read()[-2000:]
    proc.terminate()
    raise RuntimeError(f"vLLM server failed to start within 10 minutes. Last log:\n{tail}")


def stop_vllm(proc: subprocess.Popen):
    """Stop vLLM server."""
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("vLLM server stopped.")


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------
async def run_experiment(seed: int, problems: List[dict]) -> dict:
    """Run all 4 conditions for a single seed."""
    import random
    rng = random.Random(seed)
    shuffled = problems.copy()
    rng.shuffle(shuffled)

    print(f"\n{'='*60}")
    print(f"Seed {seed}: Running 4 conditions on {len(shuffled)} problems")
    print(f"Problem order: {[p['id'] for p in shuffled]}")
    print(f"{'='*60}")

    global call_counter
    results = {"seed": seed, "n_problems": len(shuffled)}

    # Condition 1: Baseline
    print("\n--- Condition 1: Baseline (CoT Pass@1) ---")
    call_counter = defaultdict(int)
    t0 = time.time()
    baseline = await run_baseline(shuffled)
    results["baseline"] = {
        "results": baseline,
        "accuracy": sum(1 for r in baseline if r["correct"]) / len(baseline),
        "calls": dict(call_counter),
        "time_s": time.time() - t0,
    }
    print(f"  Accuracy: {results['baseline']['accuracy']:.1%} ({sum(1 for r in baseline if r['correct'])}/{len(baseline)})")

    # Condition 2: Self-Consistency (N=16)
    print("\n--- Condition 2: Self-Consistency (N=16) ---")
    call_counter = defaultdict(int)
    t0 = time.time()
    sc = await run_self_consistency(shuffled, n=16)
    results["self_consistency"] = {
        "results": sc,
        "accuracy": sum(1 for r in sc if r["correct"]) / len(sc),
        "calls": dict(call_counter),
        "time_s": time.time() - t0,
    }
    print(f"  Accuracy: {results['self_consistency']['accuracy']:.1%}")

    # Condition 3: Dynamic Cheatsheet
    print("\n--- Condition 3: Dynamic Cheatsheet (N=8) ---")
    call_counter = defaultdict(int)
    t0 = time.time()
    dc = await run_dc(shuffled, n_candidates=8)
    results["dc"] = {
        "results": dc,
        "accuracy": sum(1 for r in dc if r["correct"]) / len(dc),
        "calls": dict(call_counter),
        "time_s": time.time() - t0,
    }
    print(f"  Accuracy: {results['dc']['accuracy']:.1%}")

    # Condition 4: DC + Verification
    print("\n--- Condition 4: DC + Outcome Verification ---")
    call_counter = defaultdict(int)
    t0 = time.time()
    dcv = await run_dc_verified(shuffled, n_candidates=8)
    results["dc_verified"] = {
        "results": dcv,
        "accuracy": sum(1 for r in dcv if r["correct"]) / len(dcv),
        "calls": dict(call_counter),
        "time_s": time.time() - t0,
    }
    print(f"  Accuracy: {results['dc_verified']['accuracy']:.1%}")

    return results


def main():
    parser = argparse.ArgumentParser(description="DC vs Verification on AIME 2024")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7],
                        help="Random seeds for problem ordering")
    parser.add_argument("--no-vllm", action="store_true",
                        help="Assume vLLM is already running")
    args = parser.parse_args()

    # Load data
    problems = load_aime_2024()
    print(f"Loaded {len(problems)} AIME 2024 problems")
    if len(problems) == 0:
        print("ERROR: No AIME 2024 problems found!", file=sys.stderr)
        sys.exit(1)

    # Start vLLM
    vllm_proc = None
    if not args.no_vllm:
        vllm_proc = start_vllm()
    else:
        init_clients()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        all_results = []
        for seed in args.seeds:
            result = asyncio.run(run_experiment(seed, problems))
            all_results.append(result)
            # Save incrementally
            outfile = RESULTS_DIR / f"seed_{seed}.json"
            with open(outfile, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"  Saved: {outfile}")

        # Save combined results
        combined_file = RESULTS_DIR / "all_results.json"
        with open(combined_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nAll results saved to {combined_file}")

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        conditions = ["baseline", "self_consistency", "dc", "dc_verified"]
        labels = ["Baseline (CoT)", "Self-Consistency (N=16)", "Dynamic Cheatsheet", "DC + Verification"]
        for cond, label in zip(conditions, labels):
            accs = [r[cond]["accuracy"] for r in all_results]
            mean_acc = sum(accs) / len(accs)
            std_acc = (sum((a - mean_acc) ** 2 for a in accs) / len(accs)) ** 0.5
            total_calls = sum(sum(r[cond]["calls"].values()) for r in all_results) / len(all_results)
            print(f"  {label:30s}: {mean_acc:.1%} +/- {std_acc:.1%}  (avg {total_calls:.0f} calls)")

        # Decision framework
        dc_accs = [r["dc"]["accuracy"] for r in all_results]
        dcv_accs = [r["dc_verified"]["accuracy"] for r in all_results]
        gap = (sum(dcv_accs) / len(dcv_accs)) - (sum(dc_accs) / len(dc_accs))
        print(f"\n  Gap (DC+V - DC): {gap:+.1%}")
        if gap < 0.05:
            print("  VERDICT: Verifier over-engineered → Path A (DC-only)")
        elif gap < 0.15:
            print("  VERDICT: Marginal value → Path A+ (DC + lightweight verification)")
        else:
            print("  VERDICT: Verifier justified → Path B (build verifier pipeline)")

    finally:
        if vllm_proc:
            stop_vllm(vllm_proc)


if __name__ == "__main__":
    main()
