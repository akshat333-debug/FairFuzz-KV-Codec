"""FullKV baseline runner: captures a real prefill KV cache for each variant,
generates from it losslessly (FP16, no compression), grades the answer, and
tags the intersection-full-correct subset - BEFORE any compression
evaluation, per Prompt 15 item "Run FullKV baseline first and tag the
intersection-full-correct subset before compression evaluation." Mirrors
FragKVRunner's real-capture pattern (Prompt 5) at a smaller scope (FullKV
only - no codec comparison; that is a later prompt's job).
"""

from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fairfuzzkv_codec.benchmarks.fragkv_minpairs.numeric_forms import parse_value
from fairfuzzkv_codec.benchmarks.indic_longcomp.schema import IndicGroup


@dataclass
class IndicPredictionRecord:
    group_id: str
    language: str
    task_family: str
    correct: bool
    generated_text: str
    parsed_answer: int


class IndicLongCompRunner:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32, device_map=device)
        self.model.eval()

    @torch.no_grad()
    def _generate(self, context_text: str, question_text: str, max_new_tokens: int) -> str:
        context_ids = self.tokenizer(context_text, return_tensors="pt").to(self.device)
        out = self.model(**context_ids, use_cache=True)
        cache = out.past_key_values

        suffix_ids = self.tokenizer(question_text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        out = self.model(input_ids=suffix_ids, past_key_values=cache, use_cache=True)
        next_id = torch.argmax(out.logits[:, -1, :], dim=-1)
        generated: List[int] = [int(next_id.item())]
        cur = next_id.unsqueeze(0)
        cache = out.past_key_values
        for _ in range(max_new_tokens - 1):
            out = self.model(input_ids=cur, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            cur = torch.argmax(out.logits[:, -1, :], dim=-1).unsqueeze(0)
            generated.append(int(cur.item()))
        return str(self.tokenizer.decode(generated))

    def run_group(self, group: IndicGroup, max_new_tokens: int = 4) -> List[IndicPredictionRecord]:
        records = []
        for language, variant in group.variants.items():
            text = self._generate(variant.context_text, variant.question_text, max_new_tokens)
            parsed = parse_value(text)
            records.append(IndicPredictionRecord(
                group_id=group.group_id, language=language.value, task_family=variant.task_family.value,
                correct=(parsed == group.canonical_answer), generated_text=text, parsed_answer=parsed if parsed is not None else -1,
            ))
        return records

    def run_dataset(self, groups: List[IndicGroup], max_new_tokens: int = 4) -> List[IndicPredictionRecord]:
        records = []
        for group in groups:
            records.extend(self.run_group(group, max_new_tokens))
        return records


def full_correct_group_ids(records: List[IndicPredictionRecord]) -> set:
    """Groups where EVERY language variant was answered correctly - the
    intersection-full-correct subset, computed per group (not per
    group/language), so later compression comparisons isolate compression
    effects from base-model failures on any of the 4 language conditions."""
    by_group: dict = {}
    for r in records:
        by_group.setdefault(r.group_id, []).append(r.correct)
    return {gid for gid, correct_flags in by_group.items() if all(correct_flags)}
