from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class QuantizerType(str, Enum):
    NOOP = "noop"
    UNIFORM = "uniform"
    KMEANS = "kmeans"
    VQT = "vqt"

class MetadataCoderType(str, Enum):
    NONE = "none"
    HUFFMAN = "huffman"
    ANS = "ans"
    ARITHMETIC = "arithmetic"

class ModelIdentity(BaseModel):
    model_name: str
    tokenizer_name: Optional[str] = None
    layer_count: int
    head_count: int
    head_dim: int

class LayerHeadSelection(BaseModel):
    layers: Optional[List[int]] = None  # None means all layers
    heads: Optional[List[int]] = None   # None means all heads

class CohortDefinition(BaseModel):
    type: str = "unicode" # E.g., 'unicode', 'pos', 'learned'
    params: Dict[str, str] = Field(default_factory=dict)

class CodecBudget(BaseModel):
    total_bits_per_element: float
    pruning_ratio: float = 0.0

class QuantizerConfig(BaseModel):
    quantizer_type: QuantizerType
    codebook_size: int = 256
    group_size: int = 1

class HardwareManifest(BaseModel):
    device: str
    dtype: str
    seed: int = 42

class FairFuzzKVConfig(BaseModel):
    model: ModelIdentity
    selection: LayerHeadSelection = Field(default_factory=LayerHeadSelection)
    context_length: int = 2048
    budget: CodecBudget
    quantizer: QuantizerConfig
    metadata_coder: MetadataCoderType = MetadataCoderType.NONE
    cohort: CohortDefinition = Field(default_factory=CohortDefinition)
    hardware: HardwareManifest
