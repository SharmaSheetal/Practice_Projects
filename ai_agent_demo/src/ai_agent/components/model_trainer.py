import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig as HFBitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    logging,
)
from peft import LoraConfig, get_peft_model
from ai_agent.config.configuration import ModelConfig, QLoRAConfig, BitsAndBytesConfig, TrainingConfig

class ModelTrainer:
    def __init__(self, model_config: ModelConfig, qlora_config: QLoRAConfig, 
                 bnb_config: BitsAndBytesConfig, training_config: TrainingConfig):
        self.model_config = model_config
        self.qlora_config = qlora_config
        self.bnb_config = bnb_config
        self.training_config = training_config
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def setup_quantization_config(self):
        """Setup quantization configuration"""
        compute_dtype = getattr(torch, self.bnb_config.bnb_4bit_compute_dtype)
        
        bnb_config = HFBitsAndBytesConfig(
            load_in_4bit=self.bnb_config.use_4bit,
            bnb_4bit_quant_type=self.bnb_config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.bnb_config.use_nested_quant,
        )
        
        return bnb_config, compute_dtype
    
    def load_model_and_tokenizer(self):
        """Load base model and tokenizer"""
        bnb_config, compute_dtype = self.setup_quantization_config()
        
        # Check GPU compatibility with bfloat16
        if compute_dtype == torch.float16 and self.bnb_config.use_4bit:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16: accelerate training with bf16=True")
                print("=" * 80)
        
        # Load base model
        print(f"Loading model: {self.model_config.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name,
            quantization_config=bnb_config,
            device_map=self.training_config.device_map
        )
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name, 
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        return self.model, self.tokenizer
    
    def setup_peft_model(self):
        """Setup PEFT (LoRA) configuration"""
        peft_config = LoraConfig(
            lora_alpha=self.qlora_config.lora_alpha,
            lora_dropout=self.qlora_config.lora_dropout,
            r=self.qlora_config.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Apply PEFT to model
        self.model = get_peft_model(self.model, peft_config)
        return self.model
    
    def setup_training_arguments(self):
        """Setup training arguments"""
        training_arguments = TrainingArguments(
            output_dir=self.training_config.output_dir,
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            optim=self.training_config.optim,
            save_steps=self.training_config.save_steps,
            logging_steps=self.training_config.logging_steps,
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            fp16=self.training_config.fp16,
            bf16=self.training_config.bf16,
            max_grad_norm=self.training_config.max_grad_norm,
            max_steps=self.training_config.max_steps,
            warmup_ratio=self.training_config.warmup_ratio,
            group_by_length=self.training_config.group_by_length,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            report_to="tensorboard",
            save_strategy="steps",
            eval_strategy="no",
            dataloader_drop_last=True,
            remove_unused_columns=False,
        )
        
        return training_arguments
    
    def initialize_trainer(self, train_dataset, data_collator):
        """Initialize the Hugging Face Trainer"""
        training_arguments = self.setup_training_arguments()
        
        self.trainer = Trainer(
            model=self.model,
            args=training_arguments,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        return self.trainer
    
    def train(self):
        """Train the model"""
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Call initialize_trainer() first.")
        
        print("Starting training...")
        self.trainer.train()
        
    def save_model(self):
        """Save the trained model"""
        print("Saving model...")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.training_config.output_dir)
        
        # Save only the adapter weights (for LoRA)
        self.model.save_pretrained(self.training_config.output_dir)
        self.trainer.model.save_pretrained(self.model_config.new_model)
        
        print(f"Model saved to {self.training_config.output_dir}")
        print(f"Model also saved to {self.model_config.new_model}")