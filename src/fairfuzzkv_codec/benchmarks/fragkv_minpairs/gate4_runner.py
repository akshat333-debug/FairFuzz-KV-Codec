"""Gate 4 real-model runner: captures real KV + real attention for a
FragKV-MinPairs variant, scores repair candidates with each Module 3 scorer
(fuzzy/monotone/knapsack/logistic) plus a no-repair baseline, applies the
resulting budget-neutral swap via the UNCHANGED Prompt 9 `RepairContract`,
encodes the resulting position mask through `ExplicitMaskCodec` (so every
system's bits/element is controlled by construction - see GATE4_CONFIG.md),
splices the reconstruction back into the model, and grades the generated
answer. Everything here is real, measured data - no synthetic candidates
(contrast with Prompt 13's demo, which used synthetic scorer inputs).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from fairfuzzkv_codec.benchmarks.fragkv_minpairs.numeric_forms import parse_value
from fairfuzzkv_codec.benchmarks.fragkv_minpairs.schema import MinPairGroup
from fairfuzzkv_codec.codec.baselines import FullKVFP16Codec
from fairfuzzkv_codec.codec.explicit_mask import ExplicitMaskCodec
from fairfuzzkv_codec.fragility_estimation.pipeline import compute_fragility_report
from fairfuzzkv_codec.pruning.repair import RepairContract
from fairfuzzkv_codec.repair_scoring.ablation import ScorerConfig, ScorerType, score_candidates
from fairfuzzkv_codec.repair_scoring.inputs import ScorerInputs, fit_input_normalizers, normalize_inputs
from fairfuzzkv_codec.repair_scoring.integration import propose_repair_swap
from fairfuzzkv_codec.repair_scoring.sensitivity import fuzzy_num_parameters, measure_complexity
from fairfuzzkv_codec.repair_scoring.competitors import knapsack_value_cost_ratio, logistic_score, monotone_weighted_score
from fairfuzzkv_codec.repair_scoring.fuzzy import fuzzy_priority_scores

SYSTEMS = ("no_repair", "fuzzy", "monotone", "knapsack", "logistic")
REPAIR_DELTA = 0.02
AUDITED_LAYER = 0
AUDITED_HEAD = 0  # head 0 always maps to KV-head 0 under any GQA grouping ratio


@dataclass
class Gate4PredictionRecord:
    example_id: str
    n_g: int
    budget_retention_ratio: float
    seed: int
    system: str
    correct: bool
    bits_per_element: float
    kv_mse: float
    repair_accepted: int
    repair_attempted: int


def _candidate_inputs(
    context_text: str, tokenizer: Any, attn_row: torch.Tensor
) -> ScorerInputs:
    """Real per-position candidate signals: fragility (Module 2, broadcast
    from surface group to token position), evidence_importance (real
    attention mass from the audited head), completion_cost (surface group's
    token count), staleness (positional recency). Positions not covered by
    any surface group (special tokens etc.) get neutral fragility/cost
    defaults, documented rather than silently guessed."""
    report = compute_fragility_report(context_text, tokenizer)
    seq_len = attn_row.shape[0]

    fragility = torch.zeros(seq_len)
    completion_cost = torch.ones(seq_len)
    for record, rs in zip(report.mapper_result.records, report.risk_scores):
        for idx in record.token_indices:
            if 0 <= idx < seq_len:
                fragility[idx] = rs.score
                completion_cost[idx] = float(max(record.token_count, 1))

    staleness = 1.0 - torch.arange(seq_len, dtype=torch.float32) / max(seq_len - 1, 1)

    return ScorerInputs(
        fragility=fragility, evidence_importance=attn_row.clone(),
        completion_cost=completion_cost, staleness=staleness,
    )


def _score_all_systems(inputs_norm: ScorerInputs) -> Dict[str, torch.Tensor]:
    return {
        "fuzzy": score_candidates(inputs_norm, ScorerConfig(ScorerType.FUZZY)),
        "monotone": score_candidates(inputs_norm, ScorerConfig(ScorerType.MONOTONE)),
        "knapsack": score_candidates(inputs_norm, ScorerConfig(ScorerType.KNAPSACK)),
        "logistic": score_candidates(inputs_norm, ScorerConfig(ScorerType.LOGISTIC)),
    }


def measure_scorer_latencies(inputs_norm: ScorerInputs) -> Dict[str, float]:
    reports = [
        measure_complexity("fuzzy", fuzzy_priority_scores, inputs_norm, fuzzy_num_parameters()),
        measure_complexity("monotone", monotone_weighted_score, inputs_norm, num_parameters=4),
        measure_complexity("knapsack", knapsack_value_cost_ratio, inputs_norm, num_parameters=4),
        measure_complexity("logistic", logistic_score, inputs_norm, num_parameters=5),
    ]
    return {r.scorer_name: r.latency_seconds_per_candidate for r in reports}


class Gate4Runner:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # eager attention is required for real `output_attentions=True` weights -
        # sdpa/flash-attention paths never materialize them (see Prompt 14's real
        # attention-row requirement in _capture_kv_qkv_attention).
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32, device_map=device, attn_implementation="eager"
        )
        self.model.eval()
        self.config_hash = f"gate4-study-{model_name}"

    @torch.no_grad()
    def _capture_kv_qkv_attention(self, context_text: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One forward pass: real full K/V cache (all layers/heads), real
        q/k/v for the audited (layer, head), and the real attention row
        (post-softmax) the audited head assigns from its last query position
        to every key position."""
        q_proj = self.model.get_submodule(f"model.layers.{AUDITED_LAYER}.self_attn.q_proj")
        captured: Dict[str, torch.Tensor] = {}

        def _hook(module: Any, inputs: Any, output: torch.Tensor) -> None:
            captured["q"] = output.detach()

        handle = q_proj.register_forward_hook(_hook)
        try:
            ids = self.tokenizer(context_text, return_tensors="pt").to(self.device)
            out = self.model(**ids, use_cache=True, output_attentions=True)
        finally:
            handle.remove()

        pkv = out.past_key_values
        num_layers = len(pkv.layers)
        K = torch.stack([pkv.layers[i].keys for i in range(num_layers)], dim=0)
        V = torch.stack([pkv.layers[i].values for i in range(num_layers)], dim=0)

        num_kv_heads = pkv.layers[AUDITED_LAYER].keys.shape[1]
        head_dim = pkv.layers[AUDITED_LAYER].keys.shape[-1]
        seq_len = pkv.layers[AUDITED_LAYER].keys.shape[2]

        q_full = captured["q"]  # [1, seq, num_attention_heads * head_dim]
        num_attn_heads = q_full.shape[-1] // head_dim
        q_reshaped = q_full.view(1, seq_len, num_attn_heads, head_dim).transpose(1, 2)  # [1, heads, seq, d]
        q_head = q_reshaped[0, AUDITED_HEAD]  # [seq, d]

        kv_head_idx = min(AUDITED_HEAD, num_kv_heads - 1)  # AUDITED_HEAD=0 -> kv head 0 always
        k_head = pkv.layers[AUDITED_LAYER].keys[0, kv_head_idx]  # [seq, d]
        v_head = pkv.layers[AUDITED_LAYER].values[0, kv_head_idx]  # [seq, d]

        attn = out.attentions[AUDITED_LAYER]  # [1, num_attn_heads, seq_q, seq_k], real post-softmax
        attn_row = attn[0, AUDITED_HEAD, -1, :].detach()  # last query position's real distribution

        return K, V, q_head, k_head, v_head, attn_row

    @torch.no_grad()
    def _generate_from_reconstructed_kv(self, K_recon: torch.Tensor, V_recon: torch.Tensor, suffix_ids: torch.Tensor, max_new_tokens: int) -> str:
        num_layers = K_recon.shape[0]
        kv_pairs = [(K_recon[i], V_recon[i]) for i in range(num_layers)]
        cache = DynamicCache(ddp_cache_data=kv_pairs)

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

    def run_variant(
        self, group: MinPairGroup, n_g: int, retention_ratio: float, repair_swap_fraction: float,
        seed: int, norm_stats: Any = None, max_new_tokens: int = 4,
    ) -> Tuple[List[Gate4PredictionRecord], Dict[str, float], Any]:
        variant = group.get_variant(n_g)
        K, V, q, k, v, attn_row = self._capture_kv_qkv_attention(variant.context_text)
        suffix_ids = self.tokenizer(variant.question_text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        seq_len = K.shape[3]

        raw_inputs = _candidate_inputs(variant.context_text, self.tokenizer, attn_row)
        if norm_stats is None:
            norm_stats = fit_input_normalizers(raw_inputs)
        inputs_norm = normalize_inputs(raw_inputs, norm_stats)
        scores = _score_all_systems(inputs_norm)

        keep = max(1, int(seq_len * retention_ratio))
        _, top_idx = torch.topk(attn_row, min(keep, seq_len))
        evicted0 = torch.ones(seq_len, dtype=torch.bool)
        evicted0[top_idx] = False
        num_evicted = int(evicted0.sum().item())
        swap_n = max(0, int(num_evicted * repair_swap_fraction))

        records: List[Gate4PredictionRecord] = []
        example_id = f"{group.group_id}_ng{n_g}"

        for system in SYSTEMS:
            if system == "no_repair":
                final_mask = ~evicted0
                accepted, attempted = 0, 0
            else:
                priority = scores[system]
                reintroduce, evict = propose_repair_swap(priority, evicted0, swap_n)
                attempted = 1 if reintroduce else 0
                contract = RepairContract(delta=REPAIR_DELTA)
                if reintroduce:
                    new_evicted = contract.evaluate_swap(q, k, v, evicted0, reintroduce=reintroduce, evict=evict)
                else:
                    new_evicted = evicted0
                accepted = len(contract.accepted_actions())
                final_mask = ~new_evicted

            codec = ExplicitMaskCodec(self.config_hash, final_mask)
            byte_k, meta_k = codec.encode_prefill(K)
            byte_v, meta_v = codec.encode_prefill(V)
            K_recon = codec.decode(byte_k, meta_k, tuple(meta_k["kv_shape"]), K.dtype, self.device)
            V_recon = codec.decode(byte_v, meta_v, tuple(meta_v["kv_shape"]), V.dtype, self.device)

            total_logical_bits = meta_k["accountant_report"]["logical_bits"] + meta_v["accountant_report"]["logical_bits"]
            bits_per_element = total_logical_bits / (K.numel() + V.numel())
            mse = (torch.nn.functional.mse_loss(K_recon.float(), K.float()) + torch.nn.functional.mse_loss(V_recon.float(), V.float())).item() / 2.0

            generated_text = self._generate_from_reconstructed_kv(K_recon, V_recon, suffix_ids, max_new_tokens)
            correct = parse_value(generated_text) == group.canonical_value

            records.append(Gate4PredictionRecord(
                example_id=example_id, n_g=n_g, budget_retention_ratio=retention_ratio, seed=seed,
                system=system, correct=correct,
                bits_per_element=bits_per_element, kv_mse=mse,
                repair_accepted=accepted, repair_attempted=attempted,
            ))

        # FullKV lossless control, for isolation (matches Gate 1/Gate 2 convention).
        full_codec = FullKVFP16Codec(self.config_hash)
        byte_k, meta_k = full_codec.encode_prefill(K)
        byte_v, meta_v = full_codec.encode_prefill(V)
        K_full = full_codec.decode(byte_k, meta_k, tuple(meta_k["kv_shape"]), K.dtype, self.device)
        V_full = full_codec.decode(byte_v, meta_v, tuple(meta_v["kv_shape"]), V.dtype, self.device)
        full_text = self._generate_from_reconstructed_kv(K_full, V_full, suffix_ids, max_new_tokens)
        full_correct = parse_value(full_text) == group.canonical_value
        records.append(Gate4PredictionRecord(
            example_id=example_id, n_g=n_g, budget_retention_ratio=retention_ratio, seed=seed,
            system="full", correct=full_correct,
            bits_per_element=16.0, kv_mse=0.0, repair_accepted=0, repair_attempted=0,
        ))

        latencies = measure_scorer_latencies(inputs_norm)
        return records, latencies, norm_stats
