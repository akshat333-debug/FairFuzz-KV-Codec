# Execution Gates

These are the strict Go/No-Go gates for the FairFuzzKV-Codec implementation.

## Gate 1: Baseline Parity
- **Condition**: The No-op codec must reconstruct the synthetic KV-cache exactly (bit-for-bit parity with FP16 inputs).
- **Validation**: `tests/codec/test_noop.py`

## Gate 2: Determinism
- **Condition**: Two runs with the same configuration and random seed must yield identical outputs (manifests, encoded bytes, reconstructed cache).
- **Validation**: `tests/core/test_execution.py`

## Gate 3: Budget Enforcement
- **Condition**: The exact byte accounting of the compressed stream must be mathematically less than or equal to the defined `bit_budget` in the configuration. The evaluator must refuse to return a codec when the target cannot be reached within tolerance.
- **Validation**: `tests/evaluation/test_budget.py`

## Gate 4: Manifest Integrity
- **Condition**: Every CLI command execution must write a complete, schema-validated JSONL manifest.
- **Validation**: `tests/cli/test_manifests.py`
