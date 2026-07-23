from typing import Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fairfuzzkv_codec.core.config import LayerHeadSelection

class HFCapture:
    def __init__(self, model_name: str, device: str = "cpu", dtype: torch.dtype = torch.float32):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        
        # Load model with memory-conscious settings
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map=self.device,
            low_cpu_mem_usage=True
        )
        self.model.eval()

    @torch.no_grad()
    def capture_prefill_kv(self, text: str, selection: LayerHeadSelection) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs a forward pass and captures KV tensors.
        Returns:
            K: [layers, batch, heads, seq_len, head_dim]
            V: [layers, batch, heads, seq_len, head_dim]
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Forward pass to get past_key_values
        outputs = self.model(**inputs, use_cache=True)
        past_key_values = outputs.past_key_values
        
        k_tensors = []
        v_tensors = []
        
        # Process layer selection.
        # transformers >=4.51 uses DynamicCache.layers[i].keys/.values (new API).
        # Older versions used DynamicCache.key_cache/.value_cache lists, or a
        # plain tuple-of-tuples for legacy models. Check newest API first.
        if hasattr(past_key_values, "layers"):
            num_layers = len(past_key_values.layers)
        elif hasattr(past_key_values, "key_cache"):
            num_layers = len(past_key_values.key_cache)
        elif hasattr(past_key_values, "__len__"):
            num_layers = len(past_key_values)
        else:
            num_layers = 1 # Fallback

        layer_indices = selection.layers if selection.layers is not None else list(range(num_layers))

        for layer_idx in layer_indices:
            if hasattr(past_key_values, "layers"):
                k = past_key_values.layers[layer_idx].keys
                v = past_key_values.layers[layer_idx].values
            elif hasattr(past_key_values, "key_cache"):
                k = past_key_values.key_cache[layer_idx]
                v = past_key_values.value_cache[layer_idx]
            else:
                k, v = past_key_values[layer_idx]
            
            # Process head selection
            if selection.heads is not None:
                k = k[:, selection.heads, :, :]
                v = v[:, selection.heads, :, :]
                
            k_tensors.append(k)
            v_tensors.append(v)
            
        # Stack over layers dimension: [selected_layers, batch, selected_heads, seq_len, head_dim]
        K_out = torch.stack(k_tensors, dim=0)
        V_out = torch.stack(v_tensors, dim=0)
        
        return K_out, V_out
