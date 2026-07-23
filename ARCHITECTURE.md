# FairFuzzKV-Codec Architecture

This document describes the high-level system design of the FairFuzzKV-Codec platform. It serves as the architectural contract enforcing the frozen specification.

## Core Modules

The system is split into the following functional modules:

1. **Cache Capture**: Hooks and interceptions to capture raw FP16/BF16 KV-cache activations from the underlying model.
2. **Unicode Grouping**: Analyzes the token streams to form semantic cohorts based on Unicode boundaries or other token heuristics.
3. **Fragility Estimation**: Computes the sensitivity or 'fragility' of individual tokens/heads/layers to lossy compression.
4. **Pruning**: Implements token eviction and sparse masking policies.
5. **Quantization**: Maps continuous representations to discrete codebooks based on the allocated bit budget.
6. **Allocation**: Solves the bit distribution problem across layers, heads, and cohorts based on fragility.
7. **Metadata Coding**: Compresses the non-uniform allocation maps, sparsity masks, and quantizer parameters.
8. **Decoder/Reconstruction**: Restores the approximate KV-cache from the compressed byte stream.

## Regimes

The architecture strictly separates the execution regimes:
- **Prefill**: Processes the entire prompt context in bulk. Allocation and compression happen here.
- **Decode**: Auto-regressive phase where cache is appended token-by-token. Must handle state updates and maintain the compressed representation.

## Data Flow
`Model -> Capture -> [Grouping + Fragility] -> Allocation -> [Pruning + Quantization] -> Metadata Coding -> Byte Stream`
`Byte Stream -> Decoder -> Reconstructed Cache -> Model`

## Determinism
All randomness, hardware dispatch, and configuration choices are tracked and locked via a `HardwareManifest` and strict seed control to ensure reproducibility.
