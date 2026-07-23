from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from fairfuzzkv_codec.benchmarks.fragkv_minpairs.numeric_forms import parse_value
from fairfuzzkv_codec.benchmarks.fragkv_minpairs.schema import MinPairGroup
from fairfuzzkv_codec.benchmarks.fragkv_minpairs.stats_utils import PredictionRecord
from fairfuzzkv_codec.codec.base import BaseCodec
from fairfuzzkv_codec.codec.baselines import FullKVFP16Codec, TopKCodec, UniformQuantCodec

# Both lossy baselines are matched at exactly 8.0 bits/element by construction
# (num_bits=8 and retention_ratio=0.5 each give a bits/element formula that
# doesn't depend on the data - see codec/baselines.py), so no runtime
# MatchedBitEvaluator search is needed here. FullKV (16 bits/element, fp16)
# is the lossless control, deliberately NOT matched to the lossy budget - it
# exists to check whether fragmentation confuses the base model/tokenizer
# regardless of compression (gate1.py's control_confound check).
def build_codecs(config_hash: str) -> List[Tuple[str, BaseCodec]]:
    return [
        ("FullKV", FullKVFP16Codec(config_hash)),
        ("UniformQuant8", UniformQuantCodec(config_hash, num_bits=8)),
        ("TopK50", TopKCodec(config_hash, retention_ratio=0.5)),
    ]


class FragKVRunner:
    """Executes the real causal study: captures a real prefill KV cache for
    each MinPairGroup variant, compresses+decompresses it through each
    candidate codec, splices the reconstruction back into the model to
    continue generation, and grades the result against the canonical value."""

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32, device_map=device)
        self.model.eval()
        self.config_hash = f"fragkv-study-{model_name}"

    @torch.no_grad()
    def _capture_context_kv(self, context_text: str):
        ids = self.tokenizer(context_text, return_tensors="pt").to(self.device)
        out = self.model(**ids, use_cache=True)
        pkv = out.past_key_values
        num_layers = len(pkv.layers)
        K = torch.stack([pkv.layers[i].keys for i in range(num_layers)], dim=0)
        V = torch.stack([pkv.layers[i].values for i in range(num_layers)], dim=0)
        return K, V

    @torch.no_grad()
    def _generate_from_reconstructed_kv(
        self, K_recon: torch.Tensor, V_recon: torch.Tensor, suffix_ids: torch.Tensor, max_new_tokens: int
    ) -> str:
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
        self, group: MinPairGroup, n_g: int, codecs: List[Tuple[str, BaseCodec]], max_new_tokens: int = 4
    ) -> List[PredictionRecord]:
        variant = group.get_variant(n_g)
        K, V = self._capture_context_kv(variant.context_text)
        suffix_ids = self.tokenizer(
            variant.question_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)

        records = []
        for codec_name, codec in codecs:
            byte_stream_k, meta_k = codec.encode_prefill(K)
            byte_stream_v, meta_v = codec.encode_prefill(V)
            K_recon = codec.decode(byte_stream_k, meta_k, tuple(meta_k["kv_shape"]), K.dtype, self.device)
            V_recon = codec.decode(byte_stream_v, meta_v, tuple(meta_v["kv_shape"]), V.dtype, self.device)

            total_logical_bits = meta_k["accountant_report"]["logical_bits"] + meta_v["accountant_report"]["logical_bits"]
            actual_bits_per_element = total_logical_bits / (K.numel() + V.numel())

            mse = (
                torch.nn.functional.mse_loss(K_recon.float(), K.float())
                + torch.nn.functional.mse_loss(V_recon.float(), V.float())
            ).item() / 2.0

            generated_text = self._generate_from_reconstructed_kv(K_recon, V_recon, suffix_ids, max_new_tokens)
            parsed = parse_value(generated_text)
            correct = parsed == group.canonical_value

            records.append(
                PredictionRecord(
                    group_id=group.group_id,
                    n_g=n_g,
                    codec_name=codec_name,
                    target_bits_per_element=8.0 if codec_name != "FullKV" else 16.0,
                    actual_bits_per_element=actual_bits_per_element,
                    generated_text=generated_text,
                    parsed_value=parsed,
                    correct=correct,
                    kv_reconstruction_mse=mse,
                )
            )
        return records

    def run_study(self, groups: List[MinPairGroup], max_new_tokens: int = 4) -> List[PredictionRecord]:
        codecs = build_codecs(self.config_hash)
        all_records: List[PredictionRecord] = []
        for group in groups:
            for n_g in group.variants:
                all_records.extend(self.run_variant(group, n_g, codecs, max_new_tokens=max_new_tokens))
        return all_records
