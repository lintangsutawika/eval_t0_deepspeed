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

python3 /users/zyong2/data/zyong2/mt0/data/external/eval_t0_deepspeed/run_eval_xnli.py \
--model_name_or_path /users/zyong2/data/zyong2/mt0/models/mt0_xl_t0pp \
--output_dir /users/zyong2/data/zyong2/mt0/data/processed/002 \
--dataset_name xnli \
--dataset_config_name zh \
--dataset_split_name test \
--per_device_eval_batch_size 1 \
--max_length 512 \
--predict_with_generate True \
--tokenizer_name "google/mt5-xl" \
--cache_dir "/users/zyong2/data/zyong2/huggingface"
```

| Language | T5X (T0pp_ckpt_1025K) | HF (T0pp_ckpt_1025K)
| ------------- | ------------- | ------------- |
| ar | 44 | 43.39 |
| bg | 45 | 44.77 |
| de | 47 | 46.37 |
| el | 44.5 | 44.01 |
| en | 50.16 | 49.02 |
| es | 43.57 | 43.07 |
| fr | 44 | 43.33 |
| hi | 41 | 41.06 |
| ru | 44.25 | 43.03 |
| sw | 45.2 | 44.93 |
| th | 52.4 | 51.42 |
| tr | 43 | 41.82 |
| ur | 41.8 | 41.88 |
| vi | 45.26 | 44.63 |
| zh | 46.35 | 46.31 |