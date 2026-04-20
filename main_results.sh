model_path=/tmp/Qwen2.5-VL-7B-Instruct

bash scripts/infer_eval.sh $model_path configs/qwen2_5_vl/adaretake_qwen2-5-vl_longvideobench.yaml 8 25
bash scripts/infer_eval.sh $model_path configs/qwen2_5_vl/adaretake_qwen2-5-vl_lvbench_f1024.yaml 8 25
bash scripts/infer_eval.sh $model_path configs/qwen2_5_vl/adaretake_qwen2-5-vl_mlvu.yaml 8 25
bash scripts/infer_eval.sh $model_path configs/qwen2_5_vl/adaretake_qwen2-5-vl_videomme.yaml 8 25
