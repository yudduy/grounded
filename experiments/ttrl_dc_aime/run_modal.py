"""
TTRL + ACE Co-Evolution on AIME 2024 — Modal Serverless

Migrated from ttrl_dc_aime.ipynb. Runs 4 conditions on an A100-80GB:
  1. Baseline (frozen, CoT Pass@1)
  2. ACE-only (frozen, playbook evolution, majority vote)
  3. TTRL-only (GRPO weight updates, majority vote)
  4. ACE+TTRL (co-evolution of weights + playbook)

Checkpoint-based resumability: completed phases are skipped on restart.

Usage:
    modal run experiments/ttrl_dc_aime/run_modal.py
    modal volume get ace-ttrl-results results/ --force
"""

import modal

app = modal.App("ace-ttrl-aime")

vol = modal.Volume.from_name("ace-ttrl-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "trl==0.27.2",
        "vllm",
        "transformers==4.57.3",
        "peft==0.18.1",
        "accelerate==1.12.0",
        "datasets",
        "scipy==1.15.2",
        "matplotlib==3.10.0",
        "torch",
    )
)

# ---------------------------------------------------------------------------
# All experiment code runs inside this single Modal function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=modal.gpu.A100(size="80GB"),
    volumes={"/results": vol},
    timeout=4 * 3600,
)
def run_experiment(resume: bool = True):
    import copy
    import csv
    import gc
    import json
    import os
    import re
    import time
    from abc import ABC, abstractmethod
    from collections import Counter, defaultdict
    from dataclasses import dataclass, field
    from pathlib import Path
    from typing import Any, Dict, List, Optional, Tuple

    import torch
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend (no display)
    import matplotlib.pyplot as plt

    # -------------------------------------------------------------------
    # GPU Optimizations
    # -------------------------------------------------------------------
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:True,max_split_size_mb:512"
    )

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem_gb:.1f} GB)")
    else:
        gpu_name = "CPU"
        gpu_mem_gb = 0
        print("WARNING: No GPU detected.")

    # -------------------------------------------------------------------
    # Config
    # -------------------------------------------------------------------
    @dataclass
    class Config:
        MODEL_NAME: str = "Qwen/Qwen2.5-Math-7B-Instruct"
        NUM_GENERATIONS: int = 16
        MAX_BULLETS: int = 20
        MAX_COMPLETION_LENGTH: int = 3072
        KL_COEFF: float = 0.0
        LORA_RANK: int = 64
        LORA_ALPHA: int = 64
        LORA_MODULES: List[str] = field(
            default_factory=lambda: [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
        )
        GRPO_EPOCHS: int = 20
        LR: float = 5e-6
        MAX_GRAD_NORM: float = 1.0
        BASELINE_EPISODES: int = 1
        ACE_ONLY_EPISODES: int = 5
        TRAINED_EVAL_EPISODES: int = 3
        USE_VLLM_FOR_FROZEN: bool = True
        VLLM_GPU_UTIL_FROZEN: float = 0.95
        VLLM_MAX_MODEL_LEN: int = 4096
        RESULTS_DIR: str = "/results"
        CHECKPOINTS_DIR: str = "/results/checkpoints"

    CFG = Config()

    if gpu_mem_gb < 20:
        print("Detected < 20GB GPU — reducing config for T4/L4 compatibility")
        CFG.NUM_GENERATIONS = 8
        CFG.GRPO_EPOCHS = 10
        CFG.LORA_RANK = 32
        CFG.LORA_ALPHA = 32
        CFG.VLLM_GPU_UTIL_FROZEN = 0.85

    os.makedirs(CFG.RESULTS_DIR, exist_ok=True)
    os.makedirs(CFG.CHECKPOINTS_DIR, exist_ok=True)

    print("\nConfig:")
    for k, v in vars(CFG).items():
        print(f"  {k}: {v}")

    # -------------------------------------------------------------------
    # Answer parsing
    # -------------------------------------------------------------------
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
        if is_equiv(predicted, ground_truth):
            return True
        try:
            p = int(float(predicted.replace(",", "")))
            g = int(float(ground_truth.replace(",", "")))
            return p == g
        except (ValueError, TypeError):
            return False

    # -------------------------------------------------------------------
    # Component ABCs
    # -------------------------------------------------------------------
    class Generator(ABC):
        @abstractmethod
        def generate(self, problem: str, n: int, playbook_context: str = "") -> List[Dict]:
            ...

    class Evaluator(ABC):
        @abstractmethod
        def evaluate(self, candidates: List[Dict], ground_truth: Optional[str] = None) -> Dict:
            ...

    class Curator(ABC):
        @abstractmethod
        def curate(self, playbook: Any, problem: str, solution: str,
                   is_correct: bool, reflection: str) -> Any:
            ...

    class Trainer(ABC):
        @abstractmethod
        def train_step(self, prompts: List[str], completions: List[List[str]],
                       rewards: List[List[float]]) -> Dict:
            ...

    class PlaybookManager(ABC):
        @abstractmethod
        def get_context(self) -> str:
            ...
        @abstractmethod
        def snapshot(self) -> Dict:
            ...

    class CurriculumSelector(ABC):
        @abstractmethod
        def select(self, problems: List[Dict], episode: int) -> List[Dict]:
            ...

    # -------------------------------------------------------------------
    # Data Loading: AIME 2024
    # -------------------------------------------------------------------
    AIME_CSV_URL = "https://raw.githubusercontent.com/SakanaAI/ShinkaEvolve/main/examples/adas_aime/AIME_Dataset_1983_2025.csv"
    AIME_CSV_PATH = "/tmp/AIME_Dataset_1983_2025.csv"

    def download_aime_data():
        if not os.path.exists(AIME_CSV_PATH):
            import urllib.request
            print(f"Downloading AIME dataset from {AIME_CSV_URL}...")
            try:
                urllib.request.urlretrieve(AIME_CSV_URL, AIME_CSV_PATH)
                print(f"Downloaded to {AIME_CSV_PATH}")
            except Exception as e:
                raise RuntimeError(
                    f"Could not download AIME dataset: {e}\n"
                    f"Please download manually from:\n  {AIME_CSV_URL}"
                )

    def load_aime_2024() -> List[Dict]:
        download_aime_data()
        problems = []
        with open(AIME_CSV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if str(row["Year"]).strip() == "2024":
                    problems.append({
                        "id": row["ID"],
                        "problem": row["problem"],
                        "answer": str(int(row["answer"])),
                    })
        return problems

    problems = load_aime_2024()
    print(f"Loaded {len(problems)} AIME 2024 problems")
    assert len(problems) == 30, f"Expected 30 problems, got {len(problems)}"

    # Quick parsing tests
    assert parse_answer("The answer is \\boxed{42}") == "42"
    assert parse_answer("#### 7") == "7"
    assert parse_answer("The answer is 100.") == "100"
    assert check_answer("42", "42") is True
    assert check_answer("042", "42") is True
    print("Answer parsing tests passed!")

    # -------------------------------------------------------------------
    # Condition Configs
    # -------------------------------------------------------------------
    CONDITIONS = {
        "baseline": {
            "name": "Baseline (CoT Pass@1)",
            "playbook": "null",
            "trainer": "none",
            "evaluator": "ground_truth",
            "n_generations": 1,
            "temperature": 0.0,
        },
        "ace_only": {
            "name": "ACE-only",
            "playbook": "active",
            "trainer": "none",
            "evaluator": "majority_vote",
            "n_generations": 16,
            "temperature": 0.7,
        },
        "ttrl_only": {
            "name": "TTRL-only",
            "playbook": "null",
            "trainer": "grpo",
            "evaluator": "majority_vote",
            "n_generations": 16,
            "temperature": 0.7,
        },
        "ace_ttrl": {
            "name": "ACE+TTRL",
            "playbook": "active",
            "trainer": "grpo",
            "evaluator": "majority_vote",
            "n_generations": 16,
            "temperature": 0.7,
        },
    }

    # -------------------------------------------------------------------
    # Playbook, Reflect/Curate, Evaluators
    # -------------------------------------------------------------------
    @dataclass
    class Bullet:
        id: str
        section: str
        content: str
        helpful: int = 0
        harmful: int = 0

        def to_str(self) -> str:
            return f"[{self.id}] helpful={self.helpful} harmful={self.harmful} :: {self.content}"

    class Playbook:
        def __init__(self):
            self.bullets: List[Bullet] = []
            self._next_id: int = 1

        def add(self, section: str, content: str) -> str:
            prefix = {"STRATEGIES": "str", "COMMON_MISTAKES": "err",
                       "SOLUTION_PATTERNS": "sol"}.get(section, "gen")
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

        def snapshot(self) -> Dict:
            return {
                "bullets": [
                    {"id": b.id, "section": b.section, "content": b.content,
                     "helpful": b.helpful, "harmful": b.harmful}
                    for b in self.bullets
                ],
                "next_id": self._next_id,
            }

        @classmethod
        def from_snapshot(cls, data: Dict) -> "Playbook":
            pb = cls()
            pb._next_id = data.get("next_id", 1)
            for bd in data.get("bullets", []):
                pb.bullets.append(Bullet(**bd))
            return pb

    def make_initial_playbook() -> Playbook:
        pb = Playbook()
        pb.add("STRATEGIES", "AIME problems have integer answers from 000 to 999. Always give a non-negative integer.")
        pb.add("STRATEGIES", "Break complex problems into smaller sub-problems and solve each step carefully.")
        pb.add("COMMON_MISTAKES", "Watch for off-by-one errors in counting and combinatorics problems.")
        return pb

    REFLECT_SYSTEM = (
        "You are a math reasoning analyst. Analyze the solution and whether playbook strategies helped.\n"
        'For each bullet ID used, output a JSON line: {"id": "str-00001", "tag": "helpful"}\n'
        "Tags: helpful, harmful, neutral.\n"
        "End with a brief reflection about what mathematical insight was key."
    )

    def _build_curate_system(max_bullets, current_size):
        return (
            "You are a playbook curator for math competition solving. Based on the reflection, "
            "propose operations to improve the playbook.\n"
            "Output a JSON array of operations:\n"
            '[{"op": "ADD", "section": "STRATEGIES", "content": "new insight"},\n'
            ' {"op": "UPDATE", "id": "str-00001", "content": "refined text"},\n'
            ' {"op": "DELETE", "id": "err-00002"}]\n'
            f"Sections: STRATEGIES, COMMON_MISTAKES, SOLUTION_PATTERNS\n"
            f"Max bullets: {max_bullets}. Current: {current_size}.\n"
            "Only propose operations clearly supported by the reflection. Keep it minimal."
        )

    class NullPlaybook(PlaybookManager):
        def get_context(self) -> str:
            return ""
        def snapshot(self) -> Dict:
            return {"type": "null"}
        def reflect_and_curate(self, problem, solution, is_correct, candidates, generate_fn=None):
            pass
        def batch_reflect_and_curate(self, items, batch_generate_fn=None):
            pass

    class ActivePlaybook(PlaybookManager):
        def __init__(self, generate_fn):
            self.playbook = make_initial_playbook()
            self._generate_fn = generate_fn

        def get_context(self) -> str:
            if self.playbook.size == 0:
                return ""
            return f"\nPLAYBOOK (use these strategies, reference IDs like [str-00001]):\n{self.playbook.to_str()}"

        def snapshot(self) -> Dict:
            return {"type": "active", "playbook": self.playbook.snapshot()}

        def reflect_and_curate(self, problem, solution, is_correct, candidates, generate_fn=None):
            fn = generate_fn or self._generate_fn
            best = candidates[0] if candidates else {"raw": solution, "bullets_used": []}
            bullets_used = best.get("bullets_used", [])
            raw_response = best.get("raw", solution)

            feedback = "CORRECT" if is_correct else "INCORRECT"
            bullets_text = "\n".join(
                f"  {b.to_str()}" for b in self.playbook.bullets if b.id in bullets_used
            )
            if not bullets_text:
                bullets_text = "  (none referenced)"

            reflect_user = (
                f"Problem: {problem}\n\nSolution:\n{raw_response[:2000]}\n\n"
                f"Result: {feedback}\n\nBullets referenced:\n{bullets_text}"
            )
            reflection = fn(REFLECT_SYSTEM, reflect_user, temperature=0.3, max_tokens=512)
            self._apply_reflect_tags(reflection, bullets_used, is_correct)

            pb_text = self.playbook.to_str()
            curate_system = _build_curate_system(CFG.MAX_BULLETS, self.playbook.size)
            curate_user = f"Question: {problem}\nCurrent playbook:\n{pb_text}\n\nReflection:\n{reflection}"
            curate_raw = fn(curate_system, curate_user, temperature=0.4, max_tokens=512)
            self._apply_curate_ops(curate_raw)

        def batch_reflect_and_curate(self, items: List[Dict], batch_generate_fn=None):
            if not items:
                return
            batch_fn = batch_generate_fn
            if batch_fn is None:
                return

            reflect_messages = []
            items_meta = []
            for item in items:
                best = item["candidates"][0] if item["candidates"] else {"raw": item["solution"], "bullets_used": []}
                bullets_used = best.get("bullets_used", [])
                raw_response = best.get("raw", item["solution"])
                feedback = "CORRECT" if item["is_correct"] else "INCORRECT"
                bullets_text = "\n".join(
                    f"  {b.to_str()}" for b in self.playbook.bullets if b.id in bullets_used
                )
                if not bullets_text:
                    bullets_text = "  (none referenced)"
                reflect_user = (
                    f"Problem: {item['problem']}\n\nSolution:\n{raw_response[:2000]}\n\n"
                    f"Result: {feedback}\n\nBullets referenced:\n{bullets_text}"
                )
                reflect_messages.append((REFLECT_SYSTEM, reflect_user))
                items_meta.append({"bullets_used": bullets_used, "is_correct": item["is_correct"]})

            reflections = batch_fn(reflect_messages, 0.3, 512)

            for reflection, meta in zip(reflections, items_meta):
                self._apply_reflect_tags(reflection, meta["bullets_used"], meta["is_correct"])

            pb_text = self.playbook.to_str()
            curate_system = _build_curate_system(CFG.MAX_BULLETS, self.playbook.size)
            curate_messages = []
            for item, reflection in zip(items, reflections):
                curate_user = f"Question: {item['problem']}\nCurrent playbook:\n{pb_text}\n\nReflection:\n{reflection}"
                curate_messages.append((curate_system, curate_user))

            curate_results = batch_fn(curate_messages, 0.4, 512)

            for curate_raw in curate_results:
                self._apply_curate_ops(curate_raw)

            if self.playbook.size == 0:
                old_next = self.playbook._next_id
                self.playbook = make_initial_playbook()
                self.playbook._next_id = old_next

        def _apply_reflect_tags(self, reflection: str, bullets_used: List[str], is_correct: bool):
            tags = {}
            for m in re.finditer(r'"id"\s*:\s*"([^"]+)".*?"tag"\s*:\s*"(helpful|harmful|neutral)"', reflection):
                bid, tag = m.group(1), m.group(2)
                if bid in bullets_used:
                    tags[bid] = tag
            if not tags and bullets_used:
                default_tag = "helpful" if is_correct else "harmful"
                for bid in bullets_used:
                    tags[bid] = default_tag
            for bid, tag in tags.items():
                self.playbook.tag(bid, tag)

        def _apply_curate_ops(self, curate_raw: str):
            json_match = re.search(r"\[.*\]", curate_raw, re.DOTALL)
            if not json_match:
                return
            try:
                ops = json.loads(json_match.group())
            except json.JSONDecodeError:
                return
            for op in ops:
                try:
                    if op.get("op") == "ADD" and self.playbook.size < CFG.MAX_BULLETS:
                        self.playbook.add(op.get("section", "STRATEGIES"), op.get("content", ""))
                    elif op.get("op") == "UPDATE" and op.get("id"):
                        self.playbook.update(op["id"], op.get("content", ""))
                    elif op.get("op") == "DELETE" and op.get("id"):
                        self.playbook.remove(op["id"])
                except Exception:
                    pass

    # Majority vote
    def majority_vote(answers: List[str]) -> Tuple[str, float]:
        counter = Counter()
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

    class MajorityVoteEvaluator(Evaluator):
        def evaluate(self, candidates: List[Dict], ground_truth: Optional[str] = None) -> Dict:
            answers = [c["answer"] for c in candidates]
            winner, confidence = majority_vote(answers)
            rewards = []
            for c in candidates:
                try:
                    norm = str(int(float(c["answer"].replace(",", ""))))
                except (ValueError, TypeError):
                    norm = c["answer"].strip()
                rewards.append(1.0 if norm == winner else 0.0)
            return {
                "selected_answer": winner, "confidence": confidence,
                "reward_scores": rewards,
                "metadata": {"vote_distribution": dict(Counter(answers).most_common())},
            }

    class GroundTruthEvaluator(Evaluator):
        def evaluate(self, candidates: List[Dict], ground_truth: Optional[str] = None) -> Dict:
            if not candidates:
                return {"selected_answer": "", "reward_scores": [], "metadata": {}}
            answer = candidates[0]["answer"]
            correct = check_answer(answer, ground_truth) if ground_truth else False
            return {
                "selected_answer": answer,
                "reward_scores": [1.0 if correct else 0.0],
                "metadata": {"correct": correct},
            }

    class NullTrainer(Trainer):
        def train_step(self, prompts, completions, rewards) -> Dict:
            return {"loss": 0.0, "metrics": {}}

    # -------------------------------------------------------------------
    # Model loading + GRPO setup
    # -------------------------------------------------------------------
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=CFG.LORA_RANK,
        lora_alpha=CFG.LORA_ALPHA,
        target_modules=CFG.LORA_MODULES,
        task_type="CAUSAL_LM",
        bias="none",
        use_rslora=True,
    )

    def build_system_prompt(playbook_context: str = "") -> str:
        base = (
            "You are an expert math competition solver. Solve the problem step-by-step.\n"
            "Show all your work clearly. At the end, put your final integer answer inside \\boxed{}.\n"
            "AIME answers are always integers from 0 to 999.\n"
        )
        if playbook_context:
            base += playbook_context
        return base

    def format_prompt(problem: str, playbook_context: str = "") -> str:
        system = build_system_prompt(playbook_context)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Solve this AIME problem:\n\n{problem}"},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # -------------------------------------------------------------------
    # vLLM engine management
    # -------------------------------------------------------------------
    _vllm_engine = {"llm": None}

    def init_vllm_engine():
        from vllm import LLM
        print(f"Initializing vLLM offline engine (gpu_util={CFG.VLLM_GPU_UTIL_FROZEN})...")
        t0 = time.time()
        _vllm_engine["llm"] = LLM(
            model=CFG.MODEL_NAME,
            dtype="bfloat16",
            gpu_memory_utilization=CFG.VLLM_GPU_UTIL_FROZEN,
            max_model_len=CFG.VLLM_MAX_MODEL_LEN,
            max_num_seqs=512,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            enforce_eager=True,
            seed=42,
        )
        print(f"vLLM engine ready in {time.time()-t0:.1f}s")

    def shutdown_vllm_engine():
        if _vllm_engine["llm"] is not None:
            from vllm.distributed.parallel_state import destroy_model_parallel
            del _vllm_engine["llm"]
            _vllm_engine["llm"] = None
            destroy_model_parallel()
            torch.cuda.synchronize()
            gc.collect()
            gc.collect()
            torch.cuda.empty_cache()
            print("vLLM engine shut down, GPU memory freed.")

    def vllm_batch_generate(prompts: List[str], n: int, temperature: float,
                             max_tokens: int = None) -> List[List[Dict]]:
        if max_tokens is None:
            max_tokens = CFG.MAX_COMPLETION_LENGTH
        from vllm import SamplingParams
        llm = _vllm_engine["llm"]
        assert llm is not None, "vLLM engine not initialized."

        sampling_params = SamplingParams(
            n=n,
            temperature=max(temperature, 0.01) if n > 1 else 0.01,
            max_tokens=max_tokens,
            stop=["<|endoftext|>", "<|im_end|>"],
        )
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        all_candidates = []
        for request_output in outputs:
            candidates = []
            for completion in request_output.outputs:
                text = completion.text
                answer = parse_answer(text)
                bullets_used = re.findall(r"\[((?:str|err|sol|gen)-\d{5})\]", text)
                candidates.append({
                    "answer": answer, "raw": text,
                    "bullets_used": list(set(bullets_used)),
                })
            all_candidates.append(candidates)
        return all_candidates

    def vllm_batch_text_generate(messages_list: List[Tuple[str, str]],
                                  temperature: float = 0.3,
                                  max_tokens: int = 512) -> List[str]:
        from vllm import SamplingParams
        llm = _vllm_engine["llm"]
        if llm is None:
            return ['{"id": "none", "tag": "neutral"}'] * len(messages_list)

        prompts = []
        for system, user in messages_list:
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

        sampling_params = SamplingParams(
            n=1, temperature=max(temperature, 0.01), max_tokens=max_tokens,
            stop=["<|endoftext|>", "<|im_end|>"],
        )
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        return [o.outputs[0].text.strip() for o in outputs]

    def vllm_single_generate(system: str, user: str, temperature: float = 0.3,
                              max_tokens: int = 1024) -> str:
        results = vllm_batch_text_generate([(system, user)], temperature, max_tokens)
        return results[0]

    _sync_model_ref = {"model": None}

    def sync_generate(system: str, user: str, temperature: float = 0.3, max_tokens: int = 1024) -> str:
        if _vllm_engine["llm"] is not None:
            return vllm_single_generate(system, user, temperature, max_tokens)

        model = _sync_model_ref["model"]
        if model is None:
            return '{"id": "none", "tag": "neutral"}\nNo model loaded for reflection.'
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
        inputs = inputs.to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                inputs, max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response.strip()

    # -------------------------------------------------------------------
    # GRPO reward functions
    # -------------------------------------------------------------------
    def ttrl_reward_fn(prompts, completions, **kwargs) -> list[float]:
        answers = [parse_answer(c) for c in completions]
        winner, confidence = majority_vote(answers)
        rewards = []
        for a in answers:
            try:
                norm = str(int(float(a.replace(",", ""))))
            except (ValueError, TypeError):
                norm = a.strip()
            rewards.append(1.0 if norm == winner else 0.0)
        return rewards

    _ace_ttrl_state = {
        "playbook_mgr": None,
        "problem_lookup": {},
        "episode_stats": [],
        "pending_curate": [],
    }

    def ace_ttrl_reward_fn(prompts, completions, **kwargs) -> list[float]:
        answers = [parse_answer(c) for c in completions]
        winner, confidence = majority_vote(answers)

        rewards = []
        for a in answers:
            try:
                norm = str(int(float(a.replace(",", ""))))
            except (ValueError, TypeError):
                norm = a.strip()
            rewards.append(1.0 if norm == winner else 0.0)

        pb_mgr = _ace_ttrl_state["playbook_mgr"]
        if pb_mgr is not None and prompts:
            prompt_key = prompts[0] if isinstance(prompts, list) else str(prompts)
            problem_dict = _ace_ttrl_state["problem_lookup"].get(prompt_key, None)
            ground_truth = problem_dict["answer"] if problem_dict else None
            is_correct = check_answer(winner, ground_truth) if ground_truth else False

            candidates = []
            for c_text, a in zip(completions, answers):
                bullets_used = re.findall(r"\[((?:str|err|sol|gen)-\d{5})\]", c_text)
                candidates.append({"answer": a, "raw": c_text, "bullets_used": list(set(bullets_used))})

            for c in candidates:
                for bid in c.get("bullets_used", []):
                    pb_mgr.playbook.tag(bid, "helpful" if is_correct else "harmful")

            _ace_ttrl_state["pending_curate"].append({
                "problem": problem_dict["problem"] if problem_dict else "(unknown)",
                "solution": winner,
                "is_correct": is_correct,
                "candidates": candidates,
            })
            _ace_ttrl_state["episode_stats"].append({
                "confidence": confidence, "is_correct": is_correct,
                "pb_size": pb_mgr.playbook.size if hasattr(pb_mgr, "playbook") else 0,
            })

        return rewards

    # -------------------------------------------------------------------
    # GRPO configs
    # -------------------------------------------------------------------
    grpo_config = GRPOConfig(
        output_dir=CFG.CHECKPOINTS_DIR,
        num_train_epochs=CFG.GRPO_EPOCHS,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=CFG.LR,
        max_grad_norm=CFG.MAX_GRAD_NORM,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        num_generations=CFG.NUM_GENERATIONS,
        generation_batch_size=CFG.NUM_GENERATIONS * 2,
        max_completion_length=CFG.MAX_COMPLETION_LENGTH,
        temperature=1.0,
        top_p=0.95,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.4,
        vllm_enable_sleep_mode=True,
        beta=CFG.KL_COEFF,
        report_to="none",
    )

    ace_ttrl_grpo_config = GRPOConfig(
        output_dir=CFG.CHECKPOINTS_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=CFG.LR,
        max_grad_norm=CFG.MAX_GRAD_NORM,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        num_generations=CFG.NUM_GENERATIONS,
        generation_batch_size=CFG.NUM_GENERATIONS * 2,
        max_completion_length=CFG.MAX_COMPLETION_LENGTH,
        temperature=1.0,
        top_p=0.95,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.4,
        vllm_enable_sleep_mode=True,
        beta=CFG.KL_COEFF,
        report_to="none",
    )

    print("GRPO configs ready")

    # -------------------------------------------------------------------
    # Experiment functions
    # -------------------------------------------------------------------
    from datasets import Dataset
    from transformers import TrainerState

    def run_frozen_condition(condition_key: str, problems: List[Dict],
                             num_episodes: int) -> Dict:
        cond = CONDITIONS[condition_key]
        print(f"\n{'='*60}")
        print(f"Running: {cond['name']} ({num_episodes} episodes)")
        print(f"{'='*60}")

        use_playbook = cond["playbook"] == "active"
        n_gen = cond["n_generations"]
        temp = cond["temperature"]

        if use_playbook:
            playbook_mgr = ActivePlaybook(vllm_single_generate)
        else:
            playbook_mgr = NullPlaybook()

        if cond["evaluator"] == "majority_vote":
            evaluator = MajorityVoteEvaluator()
        else:
            evaluator = GroundTruthEvaluator()

        all_episode_results = []
        playbook_snapshots = []

        for episode in range(num_episodes):
            episode_start = time.time()
            episode_correct = 0
            episode_total = 0
            episode_details = []

            playbook_context = playbook_mgr.get_context()
            prompts = [format_prompt(p["problem"], playbook_context) for p in problems]
            all_candidates = vllm_batch_generate(prompts, n=n_gen, temperature=temp)

            curate_items = []
            for p_idx, (problem, candidates) in enumerate(zip(problems, all_candidates)):
                eval_result = evaluator.evaluate(candidates, ground_truth=problem["answer"])
                selected = eval_result["selected_answer"]
                is_correct = check_answer(selected, problem["answer"])

                if use_playbook:
                    curate_items.append({
                        "problem": problem["problem"],
                        "solution": selected,
                        "is_correct": is_correct,
                        "candidates": candidates,
                    })

                episode_correct += int(is_correct)
                episode_total += 1
                episode_details.append({
                    "problem_id": problem["id"],
                    "selected_answer": selected,
                    "ground_truth": problem["answer"],
                    "correct": is_correct,
                    "n_candidates": len(candidates),
                    "confidence": eval_result.get("confidence", None),
                })

            if use_playbook and curate_items:
                playbook_mgr.batch_reflect_and_curate(curate_items, vllm_batch_text_generate)

            playbook_snapshots.append(playbook_mgr.snapshot())
            episode_acc = episode_correct / episode_total if episode_total > 0 else 0.0
            episode_time = time.time() - episode_start

            all_episode_results.append({
                "episode": episode,
                "accuracy": episode_acc,
                "correct": episode_correct,
                "total": episode_total,
                "time_s": episode_time,
                "details": episode_details,
            })

            pb_size = playbook_mgr.playbook.size if hasattr(playbook_mgr, "playbook") else 0
            print(f"  Episode {episode+1}/{num_episodes}: "
                  f"acc={episode_acc:.1%} ({episode_correct}/{episode_total}) "
                  f"pb_size={pb_size} time={episode_time:.0f}s")

        return {
            "condition": condition_key,
            "name": cond["name"],
            "episodes": all_episode_results,
            "playbook_snapshots": playbook_snapshots,
            "training_metrics": [],
        }

    def evaluate_trained_model(condition_key: str, problems: List[Dict],
                               grpo_trainer, tokenizer_ref,
                               num_episodes: int) -> Dict:
        from peft import PeftModel
        from vllm import LLM, SamplingParams

        cond = CONDITIONS[condition_key]
        print(f"\n{'='*60}")
        print(f"Evaluating: {cond['name']} (post-training, {num_episodes} episodes)")
        print(f"{'='*60}")

        n_gen = cond["n_generations"]
        temp = cond["temperature"]
        evaluator = MajorityVoteEvaluator()

        adapter_path = os.path.join(CFG.CHECKPOINTS_DIR, f"{condition_key}_adapter")
        grpo_trainer.save_model(adapter_path)
        print(f"  Adapter saved to {adapter_path}")

        print("  Merging LoRA adapter with base model (CPU)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            CFG.MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cpu",
        )
        peft_model = PeftModel.from_pretrained(base_model, adapter_path)
        merged_model = peft_model.merge_and_unload()
        merged_path = os.path.join(CFG.CHECKPOINTS_DIR, f"{condition_key}_merged")
        merged_model.save_pretrained(merged_path)
        tokenizer_ref.save_pretrained(merged_path)
        del base_model, peft_model, merged_model
        print(f"  Merged model saved to {merged_path}")

        torch.cuda.synchronize()
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()

        print("  Initializing vLLM offline engine with merged model...")
        eval_llm = LLM(
            model=merged_path,
            dtype="bfloat16",
            gpu_memory_utilization=CFG.VLLM_GPU_UTIL_FROZEN,
            max_model_len=CFG.VLLM_MAX_MODEL_LEN,
            max_num_seqs=512,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            enforce_eager=True,
            seed=42,
        )

        sampling_params = SamplingParams(
            n=n_gen,
            temperature=max(temp, 0.01),
            max_tokens=CFG.MAX_COMPLETION_LENGTH,
            stop=["<|endoftext|>", "<|im_end|>"],
        )

        all_episode_results = []

        for episode in range(num_episodes):
            episode_start = time.time()
            episode_correct = 0
            episode_total = 0
            episode_details = []

            prompts = [format_prompt(p["problem"]) for p in problems]
            outputs = eval_llm.generate(prompts, sampling_params, use_tqdm=False)

            for problem, output in zip(problems, outputs):
                candidates = []
                for completion in output.outputs:
                    text = completion.text
                    candidates.append({
                        "answer": parse_answer(text),
                        "raw": text,
                        "bullets_used": [],
                    })

                eval_result = evaluator.evaluate(candidates, ground_truth=problem["answer"])
                selected = eval_result["selected_answer"]
                is_correct = check_answer(selected, problem["answer"])

                episode_correct += int(is_correct)
                episode_total += 1
                episode_details.append({
                    "problem_id": problem["id"],
                    "selected_answer": selected,
                    "ground_truth": problem["answer"],
                    "correct": is_correct,
                    "n_candidates": len(candidates),
                    "confidence": eval_result.get("confidence", None),
                })

            episode_acc = episode_correct / episode_total if episode_total > 0 else 0.0
            episode_time = time.time() - episode_start

            all_episode_results.append({
                "episode": episode,
                "accuracy": episode_acc,
                "correct": episode_correct,
                "total": episode_total,
                "time_s": episode_time,
                "details": episode_details,
            })

            print(f"  Episode {episode+1}/{num_episodes}: "
                  f"acc={episode_acc:.1%} ({episode_correct}/{episode_total}) "
                  f"time={episode_time:.0f}s")

        from vllm.distributed.parallel_state import destroy_model_parallel
        del eval_llm
        destroy_model_parallel()
        torch.cuda.synchronize()
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "condition": condition_key,
            "name": cond["name"],
            "episodes": all_episode_results,
            "playbook_snapshots": [{"type": "null"}] * num_episodes,
            "training_metrics": [],
        }

    def reset_trainer_state(trainer):
        trainer.state = TrainerState()
        trainer.lr_scheduler = None

    def rule_based_curate(playbook_mgr, pending_items):
        pb = playbook_mgr.playbook

        to_remove = [b.id for b in pb.bullets if b.harmful > b.helpful + 2]
        for bid in to_remove:
            pb.remove(bid)

        for item in pending_items:
            if item["is_correct"]:
                best = max(item["candidates"], key=lambda c: len(c.get("bullets_used", [])))
                for bid in best.get("bullets_used", []):
                    pb.tag(bid, "helpful")

        while pb.size > CFG.MAX_BULLETS:
            worst = min(pb.bullets, key=lambda b: b.helpful / (b.helpful + b.harmful + 1))
            pb.remove(worst.id)

        if pb.size == 0:
            old_next = pb._next_id
            playbook_mgr.playbook = make_initial_playbook()
            playbook_mgr.playbook._next_id = old_next

    # -------------------------------------------------------------------
    # Checkpoint helpers
    # -------------------------------------------------------------------
    def save_checkpoint(path: str, data: dict = None):
        """Write a JSON checkpoint marker file."""
        if data is None:
            data = {"completed": True, "timestamp": time.time()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load_checkpoint(path: str) -> Optional[Dict]:
        """Load a checkpoint file if it exists."""
        p = Path(path)
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return None

    # ===================================================================
    # Main experiment with checkpoint-based resumability
    # ===================================================================
    print("=" * 60)
    print("TTRL + ACE Co-Evolution Experiment (Modal)")
    print(f"Model: {CFG.MODEL_NAME}")
    print(f"GPU: {gpu_name} ({gpu_mem_gb:.1f} GB)")
    print(f"Problems: {len(problems)} AIME 2024")
    print(f"GRPO epochs: {CFG.GRPO_EPOCHS}")
    print(f"Generations/prompt: {CFG.NUM_GENERATIONS}")
    print(f"Resume mode: {resume}")
    print("=" * 60)

    all_results = {}

    # ===================================================================
    # Phase 1: Frozen-model conditions (Baseline + ACE-only)
    # ===================================================================
    frozen_ckpt = os.path.join(CFG.RESULTS_DIR, "frozen_done.json")
    if resume and Path(frozen_ckpt).exists():
        print("\n[Phase 1] Already complete, loading from checkpoint...")
        ckpt_data = load_checkpoint(frozen_ckpt)
        # Load saved results
        with open(os.path.join(CFG.RESULTS_DIR, "baseline.json")) as f:
            all_results["baseline"] = json.load(f)
        with open(os.path.join(CFG.RESULTS_DIR, "ace_only.json")) as f:
            all_results["ace_only"] = json.load(f)
        print(f"  Baseline: {all_results['baseline']['episodes'][-1]['accuracy']:.1%}")
        print(f"  ACE-only: {all_results['ace_only']['episodes'][-1]['accuracy']:.1%}")
    else:
        print("\n[Phase 1] Initializing vLLM offline engine for frozen conditions...")
        init_vllm_engine()

        result_baseline = run_frozen_condition("baseline", problems,
                                                num_episodes=CFG.BASELINE_EPISODES)
        all_results["baseline"] = result_baseline
        with open(os.path.join(CFG.RESULTS_DIR, "baseline.json"), "w") as f:
            json.dump(result_baseline, f, indent=2, default=str)
        print(f"Baseline saved. Final acc: {result_baseline['episodes'][-1]['accuracy']:.1%}")

        result_ace = run_frozen_condition("ace_only", problems,
                                          num_episodes=CFG.ACE_ONLY_EPISODES)
        all_results["ace_only"] = result_ace
        with open(os.path.join(CFG.RESULTS_DIR, "ace_only.json"), "w") as f:
            json.dump(result_ace, f, indent=2, default=str)
        print(f"ACE-only saved. Final acc: {result_ace['episodes'][-1]['accuracy']:.1%}")

        shutdown_vllm_engine()

        save_checkpoint(frozen_ckpt, {
            "completed": True,
            "baseline_acc": result_baseline["episodes"][-1]["accuracy"],
            "ace_only_acc": result_ace["episodes"][-1]["accuracy"],
        })
        vol.commit()
        print("[Phase 1] Checkpoint saved to volume.")

    # ===================================================================
    # Phase 2: TTRL-only (GRPO training with majority-vote reward)
    # ===================================================================
    ttrl_ckpt = os.path.join(CFG.RESULTS_DIR, "ttrl_done.json")
    if resume and Path(ttrl_ckpt).exists():
        print("\n[Phase 2] Already complete, loading from checkpoint...")
        with open(os.path.join(CFG.RESULTS_DIR, "ttrl_only.json")) as f:
            all_results["ttrl_only"] = json.load(f)
        print(f"  TTRL-only: {all_results['ttrl_only']['episodes'][-1]['accuracy']:.1%}")
    else:
        print(f"\n[Phase 2] Initializing GRPOTrainer for TTRL-only ({CFG.GRPO_EPOCHS} epochs)...")

        train_prompts = [format_prompt(p["problem"]) for p in problems]
        train_dataset = Dataset.from_dict({"prompt": train_prompts})

        grpo_trainer_ttrl = GRPOTrainer(
            model=CFG.MODEL_NAME,
            reward_funcs=ttrl_reward_fn,
            args=grpo_config,
            train_dataset=train_dataset,
            peft_config=peft_config,
            processing_class=tokenizer,
        )
        print(f"GRPOTrainer initialized (TTRL-only, {grpo_config.num_train_epochs} epochs)")

        print("Starting TTRL-only training...")
        try:
            grpo_trainer_ttrl.train()
            print("TTRL-only training complete!")
        except Exception as e:
            print(f"TTRL-only training error: {e}")
            import traceback; traceback.print_exc()

        result_ttrl = evaluate_trained_model(
            "ttrl_only", problems, grpo_trainer_ttrl, tokenizer,
            num_episodes=CFG.TRAINED_EVAL_EPISODES,
        )
        all_results["ttrl_only"] = result_ttrl

        with open(os.path.join(CFG.RESULTS_DIR, "ttrl_only.json"), "w") as f:
            json.dump(result_ttrl, f, indent=2, default=str)
        print(f"TTRL-only saved. Final acc: {result_ttrl['episodes'][-1]['accuracy']:.1%}")

        grpo_trainer_ttrl.save_model(os.path.join(CFG.CHECKPOINTS_DIR, "ttrl_only_final"))

        del grpo_trainer_ttrl
        torch.cuda.empty_cache()

        save_checkpoint(ttrl_ckpt, {
            "completed": True,
            "ttrl_only_acc": result_ttrl["episodes"][-1]["accuracy"],
        })
        vol.commit()
        print("[Phase 2] Checkpoint saved to volume.")

    # ===================================================================
    # Phase 3: ACE+TTRL (co-evolution of weights + playbook)
    # ===================================================================
    ace_ttrl_ckpt = os.path.join(CFG.RESULTS_DIR, "ace_ttrl_done.json")
    if resume and Path(ace_ttrl_ckpt).exists():
        print("\n[Phase 3] Already complete, loading from checkpoint...")
        with open(os.path.join(CFG.RESULTS_DIR, "ace_ttrl.json")) as f:
            all_results["ace_ttrl"] = json.load(f)
        print(f"  ACE+TTRL: {all_results['ace_ttrl']['episodes'][-1]['accuracy']:.1%}")
    else:
        print(f"\n[Phase 3] Initializing ACE+TTRL co-evolution ({CFG.GRPO_EPOCHS} epochs)...")

        _ace_ttrl_state["playbook_mgr"] = ActivePlaybook(sync_generate)
        _ace_ttrl_state["episode_stats"] = []
        _ace_ttrl_state["pending_curate"] = []

        # Check for per-epoch checkpoint (partial Phase 3 progress)
        epoch_ckpt_path = os.path.join(CFG.RESULTS_DIR, "ace_ttrl_epoch_ckpt.json")
        start_epoch = 0
        epoch_ckpt = load_checkpoint(epoch_ckpt_path)
        if resume and epoch_ckpt is not None:
            start_epoch = epoch_ckpt["completed_epoch"] + 1
            # Restore playbook state
            if "playbook_snapshot" in epoch_ckpt:
                _ace_ttrl_state["playbook_mgr"].playbook = Playbook.from_snapshot(
                    epoch_ckpt["playbook_snapshot"]
                )
            if "episode_stats" in epoch_ckpt:
                _ace_ttrl_state["episode_stats"] = epoch_ckpt["episode_stats"]
            print(f"  Resuming from epoch {start_epoch} (of {CFG.GRPO_EPOCHS})")

        initial_pb_ctx = _ace_ttrl_state["playbook_mgr"].get_context()
        initial_prompts = [format_prompt(p["problem"], initial_pb_ctx) for p in problems]
        _ace_ttrl_state["problem_lookup"] = {
            prompt: p for prompt, p in zip(initial_prompts, problems)
        }

        grpo_trainer_ace_ttrl = GRPOTrainer(
            model=CFG.MODEL_NAME,
            reward_funcs=ace_ttrl_reward_fn,
            args=ace_ttrl_grpo_config,
            train_dataset=Dataset.from_dict({"prompt": initial_prompts}),
            peft_config=peft_config,
            processing_class=tokenizer,
        )
        _sync_model_ref["model"] = grpo_trainer_ace_ttrl.model

        # If resuming, reload LoRA weights from checkpoint
        if resume and epoch_ckpt is not None and start_epoch > 0:
            lora_ckpt_dir = os.path.join(CFG.CHECKPOINTS_DIR, "ace_ttrl_epoch_lora")
            if os.path.exists(lora_ckpt_dir):
                print(f"  Loading LoRA weights from {lora_ckpt_dir}...")
                from peft import set_peft_model_state_dict
                import safetensors.torch
                state_dict = safetensors.torch.load_file(
                    os.path.join(lora_ckpt_dir, "adapter_model.safetensors")
                )
                set_peft_model_state_dict(grpo_trainer_ace_ttrl.model, state_dict)
                print("  LoRA weights restored.")

        print(f"ACE+TTRL GRPOTrainer initialized ({CFG.GRPO_EPOCHS} manual epochs)")

        print("Starting ACE+TTRL co-evolution training...")
        ace_ttrl_epoch_metrics = epoch_ckpt.get("epoch_metrics", []) if epoch_ckpt else []
        try:
            for epoch in range(start_epoch, CFG.GRPO_EPOCHS):
                playbook_ctx = _ace_ttrl_state["playbook_mgr"].get_context()
                epoch_prompts = [format_prompt(p["problem"], playbook_ctx) for p in problems]

                grpo_trainer_ace_ttrl.train_dataset = Dataset.from_dict({"prompt": epoch_prompts})
                _ace_ttrl_state["problem_lookup"] = {
                    prompt: p for prompt, p in zip(epoch_prompts, problems)
                }
                _ace_ttrl_state["pending_curate"] = []

                reset_trainer_state(grpo_trainer_ace_ttrl)
                grpo_trainer_ace_ttrl.train()

                rule_based_curate(_ace_ttrl_state["playbook_mgr"],
                                  _ace_ttrl_state["pending_curate"])

                pb_size = _ace_ttrl_state["playbook_mgr"].playbook.size
                ace_ttrl_epoch_metrics.append({"epoch": epoch, "pb_size": pb_size})
                print(f"  ACE+TTRL epoch {epoch+1}/{CFG.GRPO_EPOCHS}: pb_size={pb_size}")

                # Per-epoch checkpoint: save playbook + LoRA weights
                lora_ckpt_dir = os.path.join(CFG.CHECKPOINTS_DIR, "ace_ttrl_epoch_lora")
                grpo_trainer_ace_ttrl.save_model(lora_ckpt_dir)
                save_checkpoint(epoch_ckpt_path, {
                    "completed_epoch": epoch,
                    "playbook_snapshot": _ace_ttrl_state["playbook_mgr"].playbook.snapshot(),
                    "episode_stats": _ace_ttrl_state["episode_stats"],
                    "epoch_metrics": ace_ttrl_epoch_metrics,
                })
                vol.commit()

            print("ACE+TTRL co-evolution training complete!")
        except Exception as e:
            print(f"ACE+TTRL training error: {e}")
            import traceback; traceback.print_exc()

        result_ace_ttrl = evaluate_trained_model(
            "ace_ttrl", problems, grpo_trainer_ace_ttrl, tokenizer,
            num_episodes=CFG.TRAINED_EVAL_EPISODES,
        )
        result_ace_ttrl["playbook_snapshots"] = [_ace_ttrl_state["playbook_mgr"].snapshot()]
        result_ace_ttrl["ace_ttrl_training_stats"] = _ace_ttrl_state["episode_stats"]
        result_ace_ttrl["ace_ttrl_epoch_metrics"] = ace_ttrl_epoch_metrics
        all_results["ace_ttrl"] = result_ace_ttrl

        with open(os.path.join(CFG.RESULTS_DIR, "ace_ttrl.json"), "w") as f:
            json.dump(result_ace_ttrl, f, indent=2, default=str)
        print(f"ACE+TTRL saved. Final acc: {result_ace_ttrl['episodes'][-1]['accuracy']:.1%}")

        grpo_trainer_ace_ttrl.save_model(os.path.join(CFG.CHECKPOINTS_DIR, "ace_ttrl_final"))
        with open(os.path.join(CFG.RESULTS_DIR, "ace_ttrl_playbook_final.json"), "w") as f:
            json.dump(_ace_ttrl_state["playbook_mgr"].snapshot(), f, indent=2)

        del grpo_trainer_ace_ttrl
        _sync_model_ref["model"] = None
        torch.cuda.empty_cache()

        save_checkpoint(ace_ttrl_ckpt, {
            "completed": True,
            "ace_ttrl_acc": result_ace_ttrl["episodes"][-1]["accuracy"],
        })
        vol.commit()
        print("[Phase 3] Checkpoint saved to volume.")

    # ===================================================================
    # Phase 4: Analysis
    # ===================================================================
    print("\n[Phase 4] Running analysis...")
    from scipy import stats

    with open(os.path.join(CFG.RESULTS_DIR, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    def bootstrap_ci(data, n_bootstrap=10000, ci=0.95, stat_fn=np.mean):
        rng = np.random.default_rng(42)
        boot_stats = []
        data = np.array(data)
        for _ in range(n_bootstrap):
            sample = rng.choice(data, size=len(data), replace=True)
            boot_stats.append(stat_fn(sample))
        boot_stats = np.array(boot_stats)
        lower = np.percentile(boot_stats, (1 - ci) / 2 * 100)
        upper = np.percentile(boot_stats, (1 + ci) / 2 * 100)
        return float(stat_fn(data)), float(lower), float(upper)

    conditions = ["baseline", "ace_only", "ttrl_only", "ace_ttrl"]
    condition_labels = {
        "baseline": "Baseline",
        "ace_only": "ACE-only",
        "ttrl_only": "TTRL-only",
        "ace_ttrl": "ACE+TTRL",
    }

    binary_outcomes = {}
    for cond in conditions:
        episodes = all_results[cond]["episodes"]
        last_ep = len(episodes) - 1
        details = episodes[last_ep]["details"]
        binary_outcomes[cond] = [d["correct"] for d in details]

    def mcnemar_test(outcomes_a, outcomes_b):
        a = np.array(outcomes_a, dtype=bool)
        b = np.array(outcomes_b, dtype=bool)
        n01 = int(np.sum(~a & b))
        n10 = int(np.sum(a & ~b))
        n = n01 + n10
        if n == 0:
            return 1.0, n01, n10
        p_value = stats.binomtest(n01, n, 0.5).pvalue
        return float(p_value), n01, n10

    print("=" * 60)
    print("Statistical Analysis")
    print("=" * 60)

    comparisons = [
        ("ace_ttrl", "ttrl_only", "ACE+TTRL vs TTRL-only"),
        ("ace_ttrl", "ace_only", "ACE+TTRL vs ACE-only"),
        ("ace_ttrl", "baseline", "ACE+TTRL vs Baseline"),
        ("ttrl_only", "baseline", "TTRL-only vs Baseline"),
        ("ace_only", "baseline", "ACE-only vs Baseline"),
    ]

    print(f"\n{'Comparison':35s} {'p-value':>10s} {'n01':>5s} {'n10':>5s} {'Sig':>5s}")
    print("-" * 65)
    for cond_a, cond_b, label in comparisons:
        p_val, n01, n10 = mcnemar_test(binary_outcomes[cond_a], binary_outcomes[cond_b])
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"{label:35s} {p_val:10.4f} {n01:5d} {n10:5d} {sig:>5s}")

    print(f"\n{'Condition':15s} {'Accuracy':>10s} {'95% CI':>20s} {'Episodes':>10s}")
    print("-" * 60)
    for cond in conditions:
        outcomes = binary_outcomes[cond]
        mean, lo, hi = bootstrap_ci(outcomes, n_bootstrap=10000)
        n_eps = len(all_results[cond]["episodes"])
        print(f"{condition_labels[cond]:15s} {mean:10.1%} [{lo:.1%}, {hi:.1%}] {n_eps:>10d}")

    ace_ttrl_outcomes = np.array(binary_outcomes["ace_ttrl"], dtype=float)
    ttrl_outcomes = np.array(binary_outcomes["ttrl_only"], dtype=float)
    diff = ace_ttrl_outcomes - ttrl_outcomes
    diff_mean, diff_lo, diff_hi = bootstrap_ci(diff, n_bootstrap=10000)
    print(f"\nACE+TTRL - TTRL-only: {diff_mean:+.1%} [{diff_lo:+.1%}, {diff_hi:+.1%}]")

    # -------------------------------------------------------------------
    # Plots (saved to volume, no plt.show())
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {"baseline": "#888888", "ace_only": "#2196F3",
              "ttrl_only": "#FF9800", "ace_ttrl": "#4CAF50"}

    # (a) Accuracy over episodes
    ax = axes[0, 0]
    for cond in conditions:
        accs = [ep["accuracy"] for ep in all_results[cond]["episodes"]]
        if len(accs) > 1:
            ax.plot(range(1, len(accs) + 1), accs, label=condition_labels[cond],
                    color=colors[cond], linewidth=2)
        else:
            ax.axhline(y=accs[0], label=f"{condition_labels[cond]} ({accs[0]:.1%})",
                        color=colors[cond], linewidth=2, linestyle="--")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Accuracy")
    ax.set_title("(a) Accuracy Over Episodes")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # (b) Playbook size over time
    ax = axes[0, 1]
    for cond in ["ace_only", "ace_ttrl"]:
        snapshots = all_results[cond].get("playbook_snapshots", [])
        sizes = []
        for snap in snapshots:
            if snap.get("type") == "null":
                sizes.append(0)
            elif "playbook" in snap:
                sizes.append(len(snap["playbook"].get("bullets", [])))
            else:
                sizes.append(0)
        if sizes:
            ax.plot(range(1, len(sizes) + 1), sizes, label=condition_labels[cond],
                    color=colors[cond], linewidth=2)
    if "ace_ttrl_epoch_metrics" in all_results.get("ace_ttrl", {}):
        metrics = all_results["ace_ttrl"]["ace_ttrl_epoch_metrics"]
        if metrics:
            sizes = [m["pb_size"] for m in metrics]
            ax.plot(range(1, len(sizes) + 1), sizes, label="ACE+TTRL (training)",
                    color=colors["ace_ttrl"], linewidth=2, linestyle="--")
    ax.set_xlabel("Episode / Epoch")
    ax.set_ylabel("Playbook Size (bullets)")
    ax.set_title("(b) Playbook Size Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (c) Reward accuracy
    ax = axes[1, 0]
    for cond in ["ace_only", "ttrl_only", "ace_ttrl"]:
        episodes = all_results[cond]["episodes"]
        reward_accs = [ep["accuracy"] for ep in episodes]
        if len(reward_accs) > 1:
            ax.plot(range(1, len(reward_accs) + 1), reward_accs, label=condition_labels[cond],
                    color=colors[cond], linewidth=2)
        else:
            ax.axhline(y=reward_accs[0], label=condition_labels[cond],
                        color=colors[cond], linewidth=2, linestyle="--")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Majority Vote Accuracy")
    ax.set_title("(c) Reward Signal Quality Over Episodes")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # (d) Final accuracy bar chart
    ax = axes[1, 1]
    x_pos = np.arange(len(conditions))
    means = []
    ci_lower = []
    ci_upper = []
    bar_colors = [colors[c] for c in conditions]

    for cond in conditions:
        outcomes = binary_outcomes[cond]
        mean, lo, hi = bootstrap_ci(outcomes, n_bootstrap=10000)
        means.append(mean)
        ci_lower.append(mean - lo)
        ci_upper.append(hi - mean)

    bars = ax.bar(x_pos, means, color=bar_colors, alpha=0.8, edgecolor="black")
    ax.errorbar(x_pos, means, yerr=[ci_lower, ci_upper], fmt="none",
                ecolor="black", capsize=5, linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([condition_labels[c] for c in conditions], rotation=15)
    ax.set_ylabel("Accuracy")
    ax.set_title("(d) Final Accuracy (Last Episode) with 95% CI")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                f"{mean:.1%}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(CFG.RESULTS_DIR, "analysis_plots.png"),
                dpi=150, bbox_inches="tight")
    print(f"Plots saved to {os.path.join(CFG.RESULTS_DIR, 'analysis_plots.png')}")

    # -------------------------------------------------------------------
    # Decision verdict
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DECISION VERDICT")
    print("=" * 60)

    ace_ttrl_acc = all_results["ace_ttrl"]["episodes"][-1]["accuracy"]
    ttrl_acc = all_results["ttrl_only"]["episodes"][-1]["accuracy"]
    ace_acc = all_results["ace_only"]["episodes"][-1]["accuracy"]
    base_acc = all_results["baseline"]["episodes"][-1]["accuracy"]

    gap = ace_ttrl_acc - ttrl_acc
    print(f"  Baseline:     {base_acc:.1%}")
    print(f"  ACE-only:     {ace_acc:.1%}")
    print(f"  TTRL-only:    {ttrl_acc:.1%}")
    print(f"  ACE+TTRL:     {ace_ttrl_acc:.1%}")
    print(f"  Gap (ACE+TTRL - TTRL-only): {gap:+.1%}")
    print()

    if gap > 0.10:
        verdict = "CO-EVOLUTION WINS"
        next_step = "Co-evolution produces synergistic improvement. Scale up: more epochs (60), larger model, add strong-model verifier."
    elif gap > 0.05:
        verdict = "MARGINAL SYNERGY"
        next_step = "Small improvement from co-evolution. Try: more epochs, verifier-filtered reward, archetype-anchored curriculum."
    elif gap > -0.05:
        verdict = "GRPO SUBSUMES PLAYBOOK"
        next_step = "Weight updates capture what the playbook provides. Focus on TTRL improvements: more data, curriculum, reward engineering."
    else:
        verdict = "PLAYBOOK INTERFERES WITH RL"
        next_step = "Playbook operations hurt training. Investigate: reward noise from reflect/curate, or playbook poisoning under RL."

    print(f"  VERDICT: {verdict}")
    print(f"  NEXT:    {next_step}")

    analysis = {
        "final_accuracies": {cond: all_results[cond]["episodes"][-1]["accuracy"] for cond in conditions},
        "gap_acettrl_minus_ttrl": gap,
        "verdict": verdict,
        "next_step": next_step,
        "grpo_epochs": CFG.GRPO_EPOCHS,
        "note": f"PoC scale ({CFG.GRPO_EPOCHS} epochs vs 60 in TTRL paper)",
    }
    with open(os.path.join(CFG.RESULTS_DIR, "analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to {os.path.join(CFG.RESULTS_DIR, 'analysis.json')}")

    vol.commit()
    print("\n[Phase 4] Final results committed to volume.")
    print("Done! Download results with: modal volume get ace-ttrl-results /results --force")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    run_experiment.remote(resume=True)
