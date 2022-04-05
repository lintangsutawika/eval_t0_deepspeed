#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
from functools import partial
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset, load_metric
import torch

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from promptsource.templates import TemplateCollection

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.12.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")

logger = logging.getLogger(__name__)

class CustomTemplate(object):
    """docstring for CustomTemplate"""
    def __init__(self, inputs_fn, targets_fn):
        super(CustomTemplate, self).__init__()
        self.inputs_fn = inputs_fn
        self.targets_fn = targets_fn

    def get_answer_choices_list(self, example):
        return None

    def apply(self, example, truncate=True, highlight_variables=False):
        inputs = self.inputs_fn(example)
        targets = self.targets_fn(example)
        return inputs, targets

# xnli_eval_mixture: List[str] = []
# xnli_langs = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]

xnli_templates = {
    "based_on_previous_passage_en": CustomTemplate(
        inputs_fn=lambda ex: "{premise} Based on the previous passage, is it true that \"{hypothesis}\"? Yes, no, or maybe?".format(**ex),
        targets_fn=lambda ex: ["Yes", "Maybe", "No"][ex["label"]]
        ),
}

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    promptsource_dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    promptsource_dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_split_name: Optional[str] = field(
        default=None, metadata={"help": "The split name of the dataset to use (via the datasets library)."}
    )
    custom_template: Optional[str] = field(
        default=None, metadata={"help": ""}
    )
    dataset_prompt: Optional[str] = field(
        default=None, metadata={"help": "The name of the prompt the dataset uses."}
    )
    custom_metric_path: Optional[str] = field(
        default=None, metadata={"help": "Path to custom metric"}
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the :obj:`decoder_start_token_id`."
            "Useful for multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token "
            "needs to be the target language token.(Usually it is the target language token)"
        },
    )

    def __post_init__(self):
        if self.dataset_name is None:
            raise ValueError("Need a dataset name")


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model.resize_token_embeddings(len(tokenizer))

    def preprocess_function(ex, prompt_fn):

        max_length = data_args.max_length
        padding = "max_length"

        result = prompt_fn(ex)
        inputs, targets = result

        #inputs = list(inputs)[0]
        #targets = list(targets)[0]

        model_inputs = tokenizer.encode_plus(
            inputs,
            add_special_tokens=False,
            padding=padding,
            max_length=max_length,
            )

        model_inputs['labels'] = tokenizer.encode(
            targets,
            add_special_tokens=False,
            padding=padding,
            max_length=258,
            )
        
        #### FIXME: 
        #### commenting out because it somehow converts the first token into -100
        # if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        #     model_inputs['labels'][model_inputs['labels'] == tokenizer.pad_token_id] = -100
        
        return model_inputs

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir
        )

        data_split = data_args.dataset_split_name
        predict_dataset = raw_dataset[data_split]

        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

        column_names = predict_dataset.column_names
        labels = predict_dataset['label'] # FIX: currently store the list of labels aside
        
        ## Get all the prompts
        # collection = TemplateCollection()
        # prompt_collections = collection.get_dataset(
        #     data_args.promptsource_dataset_name,
        #     data_args.promptsource_dataset_config_name
        # )

        # if data_args.custom_template is not None:
        #     from importlib.machinery import SourceFileLoader

        #     foo = SourceFileLoader(
        #         "map_fn",
        #         data_args.custom_template
        #         ).load_module()

        #     prompt_fn = foo.map_fn
        #     metric = load_metric(data_args.custom_metric_path)

        # else:
        #     if data_args.dataset_prompt is None:
        #         prompt = prompt_collections.all_template_names[0]
        #     else:
        #         prompt = data_args.dataset_prompt
            
        prompt_fn = xnli_templates["based_on_previous_passage_en"].apply
        # prompt_fn = prompt_collections[prompt].apply
        metric = load_metric("accuracy")

        #with training_args.main_process_first(desc="prediction dataset map pre-processing"):
        predict_dataset = predict_dataset.map(
                partial(preprocess_function, prompt_fn=prompt_fn),
                #batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    def compute_metrics(eval_preds):
        def postprocess_text(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [[label.strip()] for label in labels]

            return preds, labels

        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        if isinstance(labels, tuple):
            labels = labels[0]

        result = metric.compute(predictions=preds, references=labels)
        result = {"accuracy": result["score"]}

        # seq_acc = 100 * np.mean([p == t for p, t in zip(decoded_preds, decoded_labels)])
        # result = {"accuracy": seq_acc}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        #compute_metrics=compute_metrics,
    )

    # Evaluation
    results = {}
    max_length = data_args.max_length

    logger.info("*** Predict ***")
    torch.save(predict_dataset, os.path.join(training_args.output_dir, f"{data_args.dataset_name}_{data_args.dataset_config_name}_predict_dataset.pt"))

    predict_results = trainer.predict(
        predict_dataset,
        metric_key_prefix="predict",
    )
    torch.save(predict_results, os.path.join(training_args.output_dir, f"{data_args.dataset_name}_{data_args.dataset_config_name}_predict_results.pt"))

    metrics = predict_results.metrics
    max_predict_samples = (
        data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
    )
    metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)


    if trainer.is_world_process_zero():
        # if training_args.predict_with_generate:
        predict_results.label_ids[predict_results.label_ids == -100] = 0

        # print(tokenizer.decode(predict_results.label_ids[0]))
        # sys.exit()
        predictions = tokenizer.batch_decode(
            predict_results.predictions,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        #predictions = [pred.strip() for pred in predictions]
        predict_results.label_ids[predict_results.label_ids == -100] = 0
        label_ids = tokenizer.batch_decode(
            predict_results.label_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        #label_ids = [label.strip() for label in label_ids]

        output_prediction_file = os.path.join(training_args.output_dir, f"{data_args.dataset_name}-{data_args.dataset_config_name}-predictions.tsv")
        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            correct = total = 0
            for pred, label in zip(predictions,label_ids):
                print(pred, label, pred == label)
                if pred == label:
                    correct += 1
                total += 1
                
            writer.write(f"Accuracy: {correct/total}\n")
            writer.write("Predictions\tLabels\n")
            writer.writelines(list("{}\t{}\n".format(i,j) for i,j in zip(predictions,label_ids)))


    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

