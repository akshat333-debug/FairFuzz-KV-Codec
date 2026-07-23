import struct
import math
from typing import Dict, Any, Tuple
import torch

MAGIC_BYTES = b"FFKV"
VERSION = 1

class ByteAccountant:
    """Accounts for logical bits vs serialized size overhead."""
    def __init__(self):
        self.logical_bits = 0
        self.serialized_bytes = 0
        self.components: Dict[str, int] = {}
        
    def add_component(self, name: str, size_in_bytes: int, logical_bits: int):
        self.components[name] = size_in_bytes
        self.serialized_bytes += size_in_bytes
        self.logical_bits += logical_bits

    def add_logical_only_component(self, name: str, logical_bits: int):
        """Track a component that contributes to the theoretical logical
        bit-budget but was never actually written into the serialized byte
        stream (e.g. a baseline that stores pruned data densely instead of
        packing a real sparse mask/index list). Must never add to
        serialized_bytes, or the report would claim a file size larger than
        the actual returned bytes."""
        self.components[name] = 0
        self.logical_bits += logical_bits
        
    def report(self) -> Dict[str, Any]:
        return {
            "logical_bits": self.logical_bits,
            "serialized_bytes": self.serialized_bytes,
            "overhead_bytes": self.serialized_bytes - math.ceil(self.logical_bits / 8.0),
            "components": self.components
        }

class BinarySerializer:
    """Strict binary layout serializer for quantized/pruned KV tensors."""
    
    @staticmethod
    def serialize(config_hash: str, tensors: Dict[str, torch.Tensor], metadata: Dict[str, Any]) -> Tuple[bytes, ByteAccountant]:
        """
        Serialize tensors and metadata to a binary format.
        Tensors should be CPU tensors, properly scaled/quantized.
        """
        accountant = ByteAccountant()
        
        # 1. Header (Magic (4), Version (2), Config Hash len (1) + bytes)
        hash_bytes = config_hash.encode("utf-8")
        header = struct.pack(f"<4sHB{len(hash_bytes)}s", MAGIC_BYTES, VERSION, len(hash_bytes), hash_bytes)
        accountant.add_component("header", len(header), len(header) * 8)
        
        # 2. Metadata (JSON serialized for simplicity here, but counted as header overhead)
        import json
        meta_bytes = json.dumps(metadata).encode("utf-8")
        meta_header = struct.pack("<I", len(meta_bytes))
        accountant.add_component("metadata", len(meta_header) + len(meta_bytes), (len(meta_header) + len(meta_bytes)) * 8)
        
        # 3. Tensors
        tensor_bytes_list = []
        for name, tensor in tensors.items():
            # Name
            name_bytes = name.encode("utf-8")
            tensor_bytes_list.append(struct.pack(f"<B{len(name_bytes)}s", len(name_bytes), name_bytes))
            accountant.add_component(f"tensor_header_{name}", 1 + len(name_bytes), (1 + len(name_bytes)) * 8)
            
            # Data
            t_bytes = tensor.cpu().numpy().tobytes()
            tensor_bytes_list.append(struct.pack("<Q", len(t_bytes))) # Size
            tensor_bytes_list.append(t_bytes)
            
            # Assuming standard byte-aligned types for the baseline (int8, float16)
            # Logical bits depend on the type.
            logical_bits = tensor.numel() * (tensor.element_size() * 8)
            # If INT4, it's packed in INT8 usually, so logical bits is half. We handle that logic outside.
            if metadata.get(f"{name}_logical_bits_per_element"):
                logical_bits = int(tensor.numel() * metadata[f"{name}_logical_bits_per_element"])
                
            accountant.add_component(f"tensor_data_{name}", 8 + len(t_bytes), 64 + logical_bits)
            
        final_bytes = header + meta_header + meta_bytes + b"".join(tensor_bytes_list)
        return final_bytes, accountant

    @staticmethod
    def deserialize(data: bytes) -> Tuple[str, Dict[str, Any], Dict[str, torch.Tensor]]:
        """Deserialize the binary format back into config hash, metadata, and tensors."""
        offset = 0
        
        # Header
        magic, version, hash_len = struct.unpack_from("<4sHB", data, offset)
        offset += 7
        config_hash = struct.unpack_from(f"<{hash_len}s", data, offset)[0].decode("utf-8")
        offset += hash_len
        
        assert magic == MAGIC_BYTES, "Invalid magic bytes"
        
        # Metadata
        meta_len = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        import json
        metadata = json.loads(struct.unpack_from(f"<{meta_len}s", data, offset)[0].decode("utf-8"))
        offset += meta_len
        
        # Tensors
        tensors = {}
        while offset < len(data):
            name_len = struct.unpack_from("<B", data, offset)[0]
            offset += 1
            name = struct.unpack_from(f"<{name_len}s", data, offset)[0].decode("utf-8")
            offset += name_len
            
            t_len = struct.unpack_from("<Q", data, offset)[0]
            offset += 8
            t_bytes = data[offset:offset+t_len]
            offset += t_len
            
            # Determine shape and dtype from metadata
            shape = metadata[f"{name}_shape"]
            dtype_str = metadata[f"{name}_dtype"]
            
            import numpy as np
            np_dtype: Any
            if dtype_str == "int8":
                np_dtype = np.int8
                torch_dtype = torch.int8
            elif dtype_str == "uint8":
                np_dtype = np.uint8
                torch_dtype = torch.uint8
            elif dtype_str == "float16":
                np_dtype = np.float16
                torch_dtype = torch.float16
            elif dtype_str == "float32":
                np_dtype = np.float32
                torch_dtype = torch.float32
            else:
                raise ValueError(f"Unsupported deserialization dtype: {dtype_str}")
                
            np_arr = np.frombuffer(t_bytes, dtype=np_dtype).reshape(shape)
            tensors[name] = torch.from_numpy(np_arr.copy()).to(torch_dtype)
            
        return config_hash, metadata, tensors
