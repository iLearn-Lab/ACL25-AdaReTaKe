# 🌟 AdaReTaKe: Adaptive Redundancy Reduction for Long-Context Video-Language Understanding  
[![Paper](https://img.shields.io/badge/arXiv-2503.12559-b31b1b.svg)](https://arxiv.org/abs/2503.12559)  
*Breaking the "Memory Wall" for MLLMs with Adaptive Video Compression*

## Authors

**Xiao Wang**<sup>1,2,‡</sup>, **Qingyi Si**<sup>2,‡</sup>, **Jianlong Wu**<sup>1</sup>\*, **Shiyu Zhu**<sup>3</sup>, **Li Cao**<sup>2</sup>, **Liqiang Nie**<sup>1</sup>\*

<sup>1</sup> `Harbin Institute of Technology, Shenzhen`  
<sup>2</sup> `Huawei Technologies Co., Ltd.`  
<sup>2</sup> `Shandong University`  
‡ Equal contribution
\* Corresponding authors

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
pip install git+https://github.com/huggingface/transformers.git@f3f6c86582611976e72be054675e2bf0abb5f775
apt-get install ffmpeg  # Required for full video processing
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
# Example for VideoMME (adjust for other datasets)
bash scripts/infer_eval.sh ${Qwen2.5-VL-7B-PATH} configs/qwen2_5_vl/flexreduc_qwen2-5-vl_videomme.yaml 8
```
*Results saved in `./results`*

---

## 📄 License  
*Pending final release*  
⚠️ **Research use only** — Commercial applications require explicit permission.  

---
