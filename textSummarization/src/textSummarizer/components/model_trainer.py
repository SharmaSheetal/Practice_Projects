from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from textSummarizer.entity import ModelTrainerConfig
import torch
import os

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(str(self.config.model_ckpt))
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(str(self.config.model_ckpt)).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model_pegasus,
        )

        dataset_samsum_pt = load_from_disk(self.config.data_path)
        # trainer_args = TrainingArguments(
        #     output_dir=self.config.root_dir, num_train_epochs=1, warmup_steps=500,
        #     per_device_train_batch_size=1, per_device_eval_batch_size=1,
        #     weight_decay=0.01, logging_steps=10,
        #     evaluation_strategy='steps', eval_steps=500, save_steps=1e6,
        #     gradient_accumulation_steps=16
        # ) 
        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir, num_train_epochs=int(self.config.num_train_epochs), warmup_steps=int(self.config.warmup_steps),
            per_device_train_batch_size=int(self.config.per_device_train_batch_size), per_device_eval_batch_size=int(self.config.per_device_train_batch_size),
            weight_decay=float(self.config.weight_decay), logging_steps=int(self.config.logging_steps),
            eval_strategy=str(self.config.evaluation_strategy), eval_steps=int(self.config.eval_steps), save_steps=float(self.config.save_steps),
            gradient_accumulation_steps=int(self.config.gradient_accumulation_steps
        )
        ) 

        trainer = Trainer(
            model=model_pegasus,
            args=trainer_args,
            data_collator=seq2seq_data_collator,
            train_dataset=dataset_samsum_pt["train"].select(range(10)),  # Limiting to 1000 samples for faster training
            eval_dataset=dataset_samsum_pt["validation"].select(range(10))
                            )
        trainer.train()


        model_pegasus.save_pretrained(os.path.join(self.config.root_dir, "pegasus-samsum-model"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))