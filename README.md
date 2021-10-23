# About

This is a script to evaluate T0 on any datasets available in P3. The process uses deepspeed to help increase processing speed.

# Installation

```
pip install -r requirements.txt
```

# Usage Example
```
deepspeed --num_gpus=8 run_eval.py --deepspeed ds_config_zero3.json --model_name_or_path bigscience/T0pp --output_dir output_dir --dataset_name anli --dataset_split_name dev_r3 --per_device_eval_batch_size 1 --max_length 512 --predict_with_generate True --dataset_prompt 'can we infer'
```
