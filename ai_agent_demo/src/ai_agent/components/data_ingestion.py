import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from ai_agent.config.configuration import DataConfig
import logging

class DataIngestion:
    def __init__(self, data_config: DataConfig, tokenizer):
        self.data_config = data_config
        self.tokenizer = tokenizer
        self.dataset = None
        self.tokenized_dataset = None
        
    def load_dataset(self):
        """Load dataset from Hugging Face"""
        print(f"Loading dataset: {self.data_config.dataset_name}")
        self.dataset = load_dataset(self.data_config.dataset_name, split="train")
        print(f"Dataset loaded with {len(self.dataset)} samples")
        return self.dataset
    
    def tokenize_function(self, examples):
        """Tokenize the dataset"""
        # Adjust field name based on your dataset structure
        texts = examples["text"]  # Change this to match your dataset's text field
        
        # Tokenize with truncation and padding
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.data_config.max_seq_length,
            return_tensors="pt"
        )
        
        # For causal language modeling, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    def tokenize_dataset(self):
        """Apply tokenization to the entire dataset"""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
            
        print("Tokenizing dataset...")
        self.tokenized_dataset = self.dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=self.dataset.column_names,
            desc="Tokenizing dataset"
        )
        return self.tokenized_dataset
    
    def get_data_collator(self):
        """Get data collator for language modeling"""
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # False for causal language modeling (GPT-style)
        )
    
    def prepare_data(self):
        """Complete data preparation pipeline"""
        self.load_dataset()
        self.tokenize_dataset()
        data_collator = self.get_data_collator()
        return self.tokenized_dataset, data_collator