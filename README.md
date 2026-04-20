# 🌟 AdaReTaKe: Adaptive Redundancy Reduction for Long-Context Video-Language Understanding
[![Paper](https://img.shields.io/badge/arXiv-2503.12559-b31b1b.svg)](https://arxiv.org/abs/2503.12559)
*Breaking the "Memory Wall" for MLLMs with Adaptive Video Compression*

---

## 🤖 Reproduce with a Coding Agent (One Prompt)

Have a coding agent (Claude Code, Cursor, etc.) reproduce all paper results end-to-end with a single prompt:

```
Read AGENTS.md and reproduce the AdaReTaKe paper results end-to-end.
```

[`AGENTS.md`](AGENTS.md) contains everything the agent needs: environment setup, dataset preparation, eval commands, expected scores, and common failure modes.

<p align="center">
  <img src="misc/flexreduc_pipeline.png" alt="AdaReTaKe Framework" width="70%">
</p>

---

## 🔍 Overview  
**AdaReTaKe** is an advanced video compression framework designed for Multimodal Large Language Models (MLLMs). By adaptively reducing uneven visual redundancy across timestamps and model layers, it:  
✅ **Extends context capacity** from 256 to **2048 frames**  
✅ **Theoretically minimizes compression loss** via adaptive ratio allocation  
✅ **Outperforms SOTA** by **+2.3% (7B)** and **+2.8% (72B)** on four benchmarks  

---

## 📜 Release Note  
The current open-source version is an **early research release** capable of reproducing leaderboard results.  

🔒 **Full version status**: Undergoing internal review (commercial considerations)  
🔄 **Post-approval**: Complete code will be released for **research purposes only**  

📝 **[Request Access via Google Form](https://docs.google.com/forms/d/e/1FAIpQLSf4l6fFTJgiRawMngOBo36NpZDgpHdQuOnbUaPFObnQfZ_FRg/viewform?usp=dialog)**  
*We appreciate your interest and patience!*  

---

## 🎯 Key Contributions  
| Feature | Innovation |
|---------|------------|
| **Adaptive Redundancy Reduction** | Layer-wise + timestamp-wise compression for maximal context retention |
| **Scalability** | Validated on 7B to 72B MLLMs with consistent gains |
| **Theoretical Guarantee** | Compression ratio allocation minimizes the loss upper bound |

---

## 🛠️ Setup  

### 🌐 Environment  
```bash
# For GPU users
conda create -n retake python=3.11
pip install -r requirements.txt

# For NPU users (e.g., Ascend)
conda env create -f environment_npu.yaml

# Additional dependencies
apt-get install ffmpeg  # Required for full video processing
pip install flash-attn==2.6.3 --no-build-isolation
```

---

## 🚦 Quick Start  

### 1️⃣ Configure Paths  
Edit `demo.py`:  
```python
hf_qwen2vl7b_path = "your/local/path/to/Qwen2-VL-7B-Instruct"  
# NPU users: config_path = 'configs/demo_npu.yaml'
```

### 2️⃣ (Optional) Convert LLaVA-Video Weights  
```bash
python scripts/utils/convert_llava_video_weights_to_hf.py \
  --text_model_id /path_to/Qwen2-7B-Instruct \
  --vision_model_id /path_to/siglip-so400m-patch14-384 \
  --output_hub_path /path_to/llava-video-qwen2-7b-hf \
  --old_state_dict_id /path_to/LLaVAVideoQwen2_7B
```

### 3️⃣ Run Demo  
```bash
python demo.py
```

---

## 📈 Reproduce Results

### Dataset Preparation
- [VideoMME](docs/prepare_videomme.md)
- [MLVU](docs/prepare_mlvu.md)
- [LongVideoBench](docs/prepare_longvideobench.md)
- [LVBench](docs/prepare_lvbench.md)

### Evaluation Scripts
```bash
# Main results (paper configuration: temporal + AdaKV, 2048 frames)
bash main_results.sh

# Ablation study (1024 frames, 4 configs × 4 datasets)
bash ablation.sh
```
*Results saved in `./results`*

### Main Results (Qwen2.5-VL-7B, Paper Configuration)

| Benchmark | Frames | FPS | Score |
|-----------|--------|-----|-------|
| MLVU (M-AVG) | 2048 | 2 | 75.2 |
| LongVideoBench | 2048 | 2 | 61.6 |
| LVBench | 2048 | 2 | 50.4 |
| Video-MME | 2048 | 4 | 64.8 |

### Ablation Study: Scaling to 1024 Frames

We conduct ablation experiments at **1024 frames** (4× the 256-frame setting used in the paper) to study how each component behaves when scaling to more frames. Four configurations are compared:

| Config | Temporal | Layer Allocation | Description |
|--------|----------|-----------------|-------------|
| `no_both` | ✗ | Even | Baseline |
| `no_layer` | ✓ | Even | Temporal adaptation only |
| `no_temporal` | ✗ | AdaKV | Layer allocation only |
| `full` | ✓ | AdaKV | Full method (paper) |

**Results (overall accuracy)**:

| Config | LVBench | LongVideoBench | MLVU | VideoMME | Avg |
|--------|---------|----------------|------|----------|-----|
| Baseline (`no_both`) | 49.19 | 61.40 | 75.63 | 66.67 | 63.22 |
| Temporal only (`no_layer`) | **49.97** | 61.48 | **75.94** | 66.63 | **63.51** |
| AdaKV only (`no_temporal`) | 48.55 | 61.65 | 75.50 | 66.52 | 63.06 |
| Full (`full`) | 48.48 | **62.22** | 75.41 | 66.19 | 63.08 |

**Key observations at 1024-frame scale**:
- **Temporal adaptation remains consistently beneficial**: it improves performance on LVBench (+0.78) and MLVU (+0.31), with neutral impact on the other two benchmarks. This confirms the generalizability of the temporal adaptation mechanism.
- **Layer allocation shows dataset-dependent behavior**: AdaKV layer allocation benefits LongVideoBench (+0.82 when combined with temporal), where subtitle-rich prompts create distinct cross-modal attention patterns across layers. However, it has negative impact on LVBench (−0.64) and VideoMME (−0.48). This divergence at higher frame counts warrants further investigation — potentially through more fine-grained layer-wise budget strategies or dataset-adaptive allocation.
- **LongVideoBench is unique**: its questions include full subtitle transcripts (~3000 tokens avg), creating a fundamentally different attention landscape compared to purely visual benchmarks.

---

## 📄 License
*Pending final release*
⚠️ **Research use only** — Commercial applications require explicit permission.

---
