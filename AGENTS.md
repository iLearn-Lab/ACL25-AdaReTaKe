# AGENTS.md — AdaReTaKe Reproduction Guide for Coding Agents

This file tells any coding agent (Claude Code, Cursor, Copilot Workspace, etc.) how to
fully reproduce the AdaReTaKe paper results from scratch. Follow the sections in order.

---

## 0. One-line Quickstart (tell the agent this)

```
Read AGENTS.md and reproduce the AdaReTaKe paper results end-to-end.
```

That single prompt is enough — everything the agent needs is in this file.

---

## 1. Repo Layout

```
retake/                     # Core library (monkey-patch + KV cache)
  qwen2_5_vl.py             # Forward override for Qwen2.5-VL; chunked prefill + temporal adaptation
  qwen2_vl.py               # Forward override for Qwen2-VL (same architecture, older model)
  llava_onevision.py         # Forward override for LLaVA-OneVision
  longvideo_cache.py         # KV cache compression (StandardVidLangKVCache)
  visual_compression.py      # Visual token compression utilities
  monkeypatch.py             # Patches HuggingFace model classes at import time
  infer_eval.py              # Multi-GPU inference + evaluation entry point
  dataset_utils.py           # Dataset loaders & eval metrics

configs/qwen2_5_vl/         # One YAML per benchmark (model + data + output settings)
  adaretake_qwen2-5-vl_mlvu.yaml
  adaretake_qwen2-5-vl_longvideobench.yaml
  adaretake_qwen2-5-vl_lvbench.yaml
  adaretake_qwen2-5-vl_videomme.yaml
  flexreduc_qwen2-5-vl_*_f1024*.yaml   # Ablation configs (1024-frame setting)

dataset/                    # Benchmark annotation JSONs (symlinked; not in git)
  mlvu/mlvu.json
  longvideobench/longvideobench_val.json
  lvbench/lvbench.json
  video_mme/video_mme.json

results/                    # Output directory (symlinked; not in git)
scripts/
  infer_eval.sh             # Wrapper script: bash scripts/infer_eval.sh <model_path> <config> <n_gpus> <fps>
docs/
  prepare_*.md              # Dataset download instructions per benchmark
  exps/                     # Experiment records
```

---

## 2. Environment Setup

```bash
conda create -n retake python=3.11 -y
conda activate retake

# Install exact versions from paper
pip install -r requirements.txt

# Install flash attention
pip install flash-attn==2.6.3 --no-build-isolation

# ffmpeg required for video decoding
apt-get install -y ffmpeg
```

---

## 3. Model Weights

Download Qwen2.5-VL-7B-Instruct to a local path:

```bash
# Via HuggingFace Hub
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir /path/to/Qwen2.5-VL-7B-Instruct
```

The eval scripts reference this path as the first argument to `scripts/infer_eval.sh`.

---

## 4. Dataset Preparation

Follow the per-benchmark instructions in `docs/`:

| Benchmark | Instructions | Anno JSON expected at |
|-----------|-------------|----------------------|
| MLVU | `docs/prepare_mlvu.md` | `dataset/mlvu/mlvu.json` |
| LongVideoBench | `docs/prepare_longvideobench.md` | `dataset/longvideobench/longvideobench_val.json` |
| LVBench | `docs/prepare_lvbench.md` | `dataset/lvbench/lvbench.json` |
| Video-MME | `docs/prepare_videomme.md` | `dataset/video_mme/video_mme.json` |

---

## 5. Smoke Test (Single Case)

Before running full evals, verify the setup with the demo video:

```bash
conda activate retake
cd /path/to/video-FlexReduc

# Edit demo.py line 204: set hf_model_path = '/your/local/path/to/Qwen2.5-VL-7B-Instruct'
python demo.py
```

**Expected output** — all 3 questions answered correctly:
```
Q1: A  (rabbit-human relationship)
Q2: B  (video impression)
Q3: C  (video subject)
```

If any answer is wrong, check environment and model path before proceeding.

---

## 6. Full Benchmark Evaluation (Paper Reproduction)

Run each benchmark sequentially using `main_results.sh`:

```bash
# Edit model_path in main_results.sh, then:
bash main_results.sh
```

Or run individually:

```bash
MODEL=/path/to/Qwen2.5-VL-7B-Instruct

bash scripts/infer_eval.sh $MODEL configs/qwen2_5_vl/adaretake_qwen2-5-vl_longvideobench.yaml 8 25
bash scripts/infer_eval.sh $MODEL configs/qwen2_5_vl/adaretake_qwen2-5-vl_lvbench.yaml 8 25
bash scripts/infer_eval.sh $MODEL configs/qwen2_5_vl/adaretake_qwen2-5-vl_mlvu.yaml 8 25
bash scripts/infer_eval.sh $MODEL configs/qwen2_5_vl/adaretake_qwen2-5-vl_videomme.yaml 8 25
```

> **Important**: Run benchmarks one at a time. The distributed backend uses a hardcoded
> port (12355 in `retake/infer_eval.py`). Parallel runs on the same machine will fail
> with `EADDRINUSE`.

Results are written to `results/` as `eval_results.csv` per benchmark.

---

## 7. Expected Results (Qwen2.5-VL-7B, AdaReTaKe Full Method)

Paper configuration: temporal adaptation + AdaKV layer allocation, 2048 frames (LVBench uses 1024).

| Benchmark | Frames | FPS | Metric | Score |
|-----------|--------|-----|--------|-------|
| MLVU | 2048 | 2 | M-AVG | 75.2% |
| LongVideoBench | 2048 | 2 | overall | 61.6% |
| LVBench | 1024 | 2 | overall | 48.5% |
| Video-MME | 2048 | 4 | overall | 64.8% |

**Note**: LongVideoBench includes full subtitle text in each question (~3000 tokens avg),
which is truncated to 152 tokens (`max_guide_length`) for compression guidance.

---

## 8. Key Implementation Details

### Chunked Prefill
Long videos are processed in chunks to avoid OOM. Controlled by:
- `chunked_prefill_frames: 32` — frames per chunk fed into the LLM
- `frame_chunk_size: 64` — frames per chunk fed into the visual encoder

### KV Cache Compression (`StandardVidLangKVCache`)
After each chunk's prefill, `after_forward()` selects the top-k tokens by attention score
and discards the rest. Compression ratio is controlled by:
- `compression_ratio` — global budget (derived from `max_input_length`)
- `budget_allocation_method: adakv` — allocates more budget to layers with higher variance
- `prompt_guided_compression: True` — uses the question tokens to guide which visual tokens to keep

### Temporal Adaptation (AdaReTaKe Eq. 9)
Implemented in `compute_temporal_adaptation_ratios()` (`retake/qwen2_5_vl.py`).
Called before the chunked prefill loop. For each chunk:
1. Visual embeddings are averaged over spatial patches → one vector per frame
2. Frames within each chunk are averaged → one vector per chunk
3. Each chunk's diversity score = average cosine distance to its left/right neighbors
4. Scores are normalized so the mean ratio = 1.0 (budget-neutral redistribution)

Higher diversity → more tokens kept. Boundary chunks use padding (copy of first/last chunk).

### Per-Dataset Config Differences
| Dataset | `pos_embed_reforge` | `max_guide_length` | Notes |
|---------|--------------------|--------------------|-------|
| LVBench | True | — | — |
| VideoMME | True | — | Uses 4 FPS |
| LongVideoBench | False | 152 | Subtitle text in prompts |
| MLVU | False | — | — |

---

## 9. Common Failure Modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `EADDRINUSE` on port 12355 | Another eval process still running | Wait for previous job to finish; run sequentially |
| `n_video_tokens != n_video_features` | Mismatch in video token count | Check `video_grid_thw` and `inputs_embeds` alignment |
| OOM during visual encoder | Too many frames in one batch | Lower `frame_chunk_size` in config |
| OOM during LLM prefill | Chunk size too large | Lower `chunked_prefill_frames` in config |
| G-AVG = 0.0 in MLVU | `openai` not installed | Install `openai` package for generative task scoring |
| `AssertionError` in `longvideo_cache.py` | Temporal adaptation ratio ≥ 1.0 triggers text-prefill branch | Already fixed in current code |

---

## 10. Codebase Conventions

- **Monkey-patching**: `retake/monkeypatch.py` replaces specific methods of HuggingFace
  model classes at runtime. The original class is never modified on disk.
- **Config-driven**: All hyperparameters live in YAML configs under `configs/`. No magic
  numbers in core code.
- **Single-video assumption**: The forward function assumes one video segment per input.
  Multi-video inputs are not supported.
- **Conda env**: Always run inside `conda activate retake`. The system Python may use a
  different transformers version which is incompatible.
- **transformers >= 4.57 compatibility**: The code handles both old and new architecture
  layouts (e.g., `self.visual` vs `self.model.visual`) via version-aware shimming.
