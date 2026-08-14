"""
LLM Judge inference pipeline.
Supports local HuggingFace models with configurable precision.
"""

import time
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class JudgeConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    torch_dtype: str = "float16"
    max_new_tokens: int = 64
    temperature: float = 0.01
    do_sample: bool = False
    device: str = "cuda"
    load_in_4bit: bool = False


@dataclass
class JudgeResult:
    verdict: str
    raw_output: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    metadata: Dict = field(default_factory=dict)


class LLMJudge:
    def __init__(self, config: JudgeConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        print(f"[Judge] Loading model: {self.config.model_name}")
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.config.torch_dtype, torch.float16)

        kwargs = {
            "torch_dtype": dtype,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name, **kwargs
        )
        self.model.eval()
        print(f"[Judge] Model loaded on {self.config.device}")

    def judge(self, system_prompt: str, user_prompt: str,
              metadata: Optional[Dict] = None) -> JudgeResult:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature if self.config.do_sample else None,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        new_tokens = outputs[0][prompt_tokens:]
        raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        completion_tokens = len(new_tokens)

        verdict = self._parse_verdict(raw_output)

        return JudgeResult(
            verdict=verdict,
            raw_output=raw_output,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=elapsed_ms,
            metadata=metadata or {},
        )

    def judge_batch(self, tasks: List[Tuple[str, str, Dict]]) -> List[JudgeResult]:
        """Sequential batch processing (single GPU, no padding complexity)."""
        results = []
        for i, (sys_prompt, user_prompt, meta) in enumerate(tasks):
            result = self.judge(sys_prompt, user_prompt, meta)
            results.append(result)
            if (i + 1) % 20 == 0:
                print(f"[Judge] Processed {i+1}/{len(tasks)} tasks")
        return results

    @staticmethod
    def _parse_verdict(raw: str) -> str:
        raw_upper = raw.strip().upper()

        if raw_upper in ("A", "B", "TIE"):
            return raw_upper

        for rating in ["5", "4", "3", "2", "1"]:
            if raw_upper.startswith(rating):
                return rating

        verdict_match = re.search(r'VERDICT:\s*([ABab]|TIE|tie)', raw, re.IGNORECASE)
        if verdict_match:
            return verdict_match.group(1).upper()

        if "list a" in raw.lower() and "list b" not in raw.lower():
            return "A"
        if "list b" in raw.lower() and "list a" not in raw.lower():
            return "B"

        for ch in raw_upper:
            if ch in ("A", "B"):
                return ch
            if ch in "12345":
                return ch

        return "UNPARSEABLE"

    def cleanup(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()
