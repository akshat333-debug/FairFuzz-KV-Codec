# Reproducibility Guide & Checklist

Written for someone with **no prior knowledge of this repository** on a fresh
machine. Every command below was executed on the release commit; where a step
needs network or minutes of compute, that is stated instead of hidden.

## 0. Requirements

| Requirement | Value |
|---|---|
| OS | Linux or macOS (developed on macOS arm64; CI runs ubuntu-latest) |
| Python | >= 3.12 (pinned by `.python-version`) |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| Disk | ~6 GB (torch + two model checkpoints) |
| Network | Needed once, for dependencies and Hugging Face model downloads |
| GPU | **Not required.** Everything here runs on CPU |

## 1. Fresh-machine install (one command)

```bash
git clone https://github.com/akshat333-debug/FairFuzz-KV-Codec.git
cd FairFuzz-KV-Codec
uv sync
```

`uv sync` resolves from the committed `uv.lock`, so dependency versions are
pinned exactly.

## 2. Verify the release without a model download

These use only committed artifacts — no network, no GPU, ~3 minutes:

```bash
uv run pytest -q                                   # full test suite
uv run ruff check . && uv run mypy .               # lint + types
uv run python scripts/run_release_checklist.py --skip-install
```

The checklist prints PASS/FAIL per item and writes
`release/release_checklist.json`. **SKIPPED is not PASS** — items needing a real
model are marked SKIPPED unless you pass `--full`.

### Reproduce the gate decisions from raw predictions (no model needed)

This is the strongest reproducibility property in the project: **the gate
decisions recompute from committed raw predictions alone.**

```bash
uv run python -c "
from scripts.run_gate1_study import compute_gate1_from_predictions
print('Gate 1:', compute_gate1_from_predictions('gate1_study/predictions.jsonl').decision.value)"
# expected: Gate 1: WEAK_PASS

uv run python -c "
from fairfuzzkv_codec.evaluation.gate4 import compute_gate4_from_predictions
print('Gate 4:', compute_gate4_from_predictions('gate4_fairness_study/predictions.jsonl').decision.value)"
# expected: Gate 4: FAIL
```

### Verify you received the exact released artifacts

```bash
shasum -a 256 -c release/CHECKSUMS.sha256      # macOS
sha256sum -c release/CHECKSUMS.sha256          # Linux
```

### Decode a sample bitstream (format check, no model)

```bash
uv run python -c "
from fairfuzzkv_codec.decoder import decode_from_container
blob = open('release/sample_bitstreams/scalar_int8.ffkv','rb').read()
t, r = decode_from_container(blob)
print('shape', tuple(t.shape), '| shape_ok', r.shape_ok, '| codec', r.codec_name)"
```

## 3. Course subset — the graded reproduction path

Downloads `Qwen/Qwen2.5-0.5B` (~1 GB) on first run.

```bash
# 3a. Grade-floor smoke: real capture -> encode -> decode -> matched-bit eval
uv run python scripts/demo.py

# 3b. Regenerate the IndicLongComp course + journal datasets (structural
#     validation, fragility distributions, real FullKV run on the course subset)
uv run python scripts/run_indiclongcomp_study.py

# 3c. Systems profile -> systems_profile/ (see PERFORMANCE.md)
uv run python scripts/run_systems_profile.py

# 3d. Dashboard
uv run streamlit run dashboard_app.py
```

Expected wall-clock on a 4-thread CPU laptop: 3a ≈ 1 min, 3b ≈ 6 min,
3c ≈ 3 min. First run adds model-download time.

## 4. Full study reproduction (hours, optional)

```bash
uv run python scripts/run_gate1_study.py --num-groups 200   # ~2400 generations
uv run python scripts/run_gate2_study.py --budgets 5,6,7 --seeds 42,7
uv run python scripts/run_gate4_study.py
uv run python scripts/run_gate3_study.py                    # + TinyLlama (~2 GB)
uv run python scripts/run_baseline_matrix.py
uv run python scripts/run_quantization_benchmark.py
uv run python scripts/run_lbg_benchmark.py
uv run python scripts/run_allocation_demo.py
uv run python scripts/run_minimax_demo.py
uv run python scripts/run_gate3_interactions.py
uv run python scripts/run_repair_scoring_demo.py
uv run python scripts/run_cross_tokenizer_stability.py
```

## 5. Docker

```bash
docker build -t fairfuzzkv-codec .
docker run --rm fairfuzzkv-codec inspect
```

## Reproducibility checklist

| # | Item | How it is guaranteed | Status |
|---|---|---|---|
| 1 | Pinned dependencies | `uv.lock` committed; `uv sync --frozen` in CI and checklist | Yes |
| 2 | Pinned Python | `.python-version`, `requires-python >=3.12` | Yes |
| 3 | Deterministic seeds | `ExecutionManager` seed control; every study takes `--seed` | Yes |
| 4 | Config hashing | `compute_config_hash`; run manifests written per run | Yes |
| 5 | Immutable dataset splits | sha256 `split_hash` in every dataset card | Yes |
| 6 | Raw predictions retained | `*_study/predictions.jsonl` committed for every gate | Yes |
| 7 | Gate decisions recomputable without a model | `compute_gate1_from_predictions`, `compute_gate4_from_predictions` | Yes |
| 8 | Pre-registered thresholds | frozen in `gate*.py`, unit-tested **before** each real run | Yes |
| 9 | Byte-exact format | `FORMAT.md` + golden vectors + CRC32 + fuzz tests | Yes |
| 10 | Artifact integrity | `release/CHECKSUMS.sha256` | Yes |
| 11 | Hardware manifest with results | captured in `systems_profile/`, incl. power mode | Yes |
| 12 | Measured vs estimated separated | `LatencyStats.measured`; `speedup()` refuses overlapping CIs | Yes |
| 13 | Negative results preserved | Gate 2/Gate 4 FAILs in README, ledger, dashboard, report | Yes |
| 14 | Provenance in every artifact | git commit recorded in baseline cards + experiment registry | Yes |
| 15 | CI enforcement | ruff, mypy, pytest, perf-regression thresholds | Yes |
| 16 | Independent verification path | `scripts/run_release_checklist.py` | Yes |

## Known reproduction caveats

- **Model downloads need network.** Offline machines can still run everything in
  §2, including both gate-decision reproductions.
- **Exact latency numbers will differ** across hardware. `PERFORMANCE.md` ships
  the hardware manifest and power mode so a difference is interpretable; the
  *relative* findings (prefill ≫ encode; LBG ≫ scalar encode cost) are the
  reproducible part.
- **`scripts/run_gate1_study.py` at 200 groups takes ~1–2 hours on CPU.** The
  committed `predictions.jsonl` was re-run from scratch and matched the original
  exactly, so the fast reproduction in §2 is sufficient for verifying the
  decision.
- **TinyLlama is ~2.2 GB.** Gate 3 will download it on first run.
