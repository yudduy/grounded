"""Experiment configuration for Memory x RL interaction study."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ExperimentConfig:
    # Model
    MODEL_NAME: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

    # Generation
    NUM_GENERATIONS: int = 16
    MAX_COMPLETION_LENGTH: int = 4096
    TEMPERATURE: float = 0.7

    # Playbook
    MAX_BULLETS: int = 20

    # GRPO Training
    KL_COEFF: float = 0.01
    LORA_RANK: int = 64
    LORA_ALPHA: int = 64
    LORA_MODULES: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    GRPO_EPOCHS: int = 20
    LR: float = 5e-6
    MAX_GRAD_NORM: float = 0.5
    PER_DEVICE_BATCH_SIZE: int = 2
    GRADIENT_ACCUMULATION_STEPS: int = 2

    # Seeds
    SEEDS: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])

    # Conditions
    TRAINING_CONDITIONS: List[str] = field(default_factory=lambda: ["B", "C", "D", "E"])
    EVAL_ONLY_CONDITIONS: List[str] = field(default_factory=lambda: ["C-abs", "D-abs"])

    # vLLM
    VLLM_GPU_UTIL_FROZEN: float = 0.95
    VLLM_GPU_UTIL_COLOCATE: float = 0.4
    VLLM_MAX_MODEL_LEN: int = 4096

    # Paths
    RESULTS_DIR: str = "/results"
    CHECKPOINTS_DIR: str = "/results/checkpoints"

    # Kimi API
    KIMI_REFLECT_MODEL: str = "moonshot-v1-128k"
    KIMI_CLASSIFY_MODEL: str = "moonshot-v1-8k"

    # Novelty reward
    NOVELTY_WEIGHT: float = 0.3
