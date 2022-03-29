# About

This is a script to evaluate T0 on any datasets available in P3. The process uses deepspeed to help increase processing speed.

# Installation

```
pip install -r requirements.txt
```

# Usage Example (XNLI with mT0)
```
# module load cuda/11.1.1
# module load gcc/10.2

deepspeed --num_gpus=1 run_eval.py --deepspeed ds_config_zero3.json --model_name_or_path /users/zyong2/data/zyong2/mt0/models/mt0_xl_t0pp --output_dir /users/zyong2/data/zyong2/mt0/data/processed/002 --dataset_name xnli --dataset_config_name ar --dataset_split_name test --per_device_eval_batch_size 1 --max_length 512 --predict_with_generate True --tokenizer_name "google/mt5-xl" --cache_dir "/users/zyong2/data/zyong2/huggingface" --promptsource_dataset_name anli
```
