#!/bin/bash
# Ablation study: 4 configs × 4 datasets at 1024 frames, 2 FPS
#
# Configs:
#   no_both     — baseline (no temporal, even layer allocation)
#   no_layer    — temporal adaptation only (even layer allocation)
#   no_temporal — AdaKV layer allocation only (no temporal)
#   full        — full method (temporal + AdaKV)
#
# Usage:
#   1. Set model_path below to your local Qwen2.5-VL-7B-Instruct path
#   2. bash ablation.sh
#
# Results are saved to results/qwen25vl_7b_{dataset}_f1024_2fps_r448/

model_path=/path/to/Qwen2.5-VL-7B-Instruct
N_GPUS=8
FPS=25

echo "=========================================="
echo "Ablation: LVBench (reforge=True)"
echo "=========================================="
for cfg in no_both no_layer full no_temporal; do
    suffix="_${cfg}"
    [ "$cfg" = "full" ] && suffix=""
    bash scripts/infer_eval.sh $model_path configs/qwen2_5_vl/adaretake_qwen2-5-vl_lvbench_f1024${suffix}.yaml $N_GPUS $FPS
done

echo "=========================================="
echo "Ablation: LongVideoBench (reforge=False)"
echo "=========================================="
for cfg in no_both no_layer full no_temporal; do
    suffix="_${cfg}"
    [ "$cfg" = "full" ] && suffix=""
    bash scripts/infer_eval.sh $model_path configs/qwen2_5_vl/adaretake_qwen2-5-vl_longvideobench_f1024${suffix}.yaml $N_GPUS $FPS
done

echo "=========================================="
echo "Ablation: MLVU (reforge=False)"
echo "=========================================="
for cfg in no_both no_layer full no_temporal; do
    suffix="_${cfg}"
    [ "$cfg" = "full" ] && suffix=""
    bash scripts/infer_eval.sh $model_path configs/qwen2_5_vl/adaretake_qwen2-5-vl_mlvu_f1024${suffix}.yaml $N_GPUS $FPS
done

echo "=========================================="
echo "Ablation: VideoMME (reforge=True)"
echo "=========================================="
for cfg in no_both no_layer full no_temporal; do
    suffix="_${cfg}"
    [ "$cfg" = "full" ] && suffix=""
    bash scripts/infer_eval.sh $model_path configs/qwen2_5_vl/adaretake_qwen2-5-vl_videomme_f1024${suffix}.yaml $N_GPUS $FPS
done

echo "=========================================="
echo "All ablation experiments complete!"
echo "=========================================="
