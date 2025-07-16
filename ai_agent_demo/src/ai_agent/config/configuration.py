import os
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for model parameters"""
    model_name: str = "NousResearch/Llama-2-7b-chat-hf"
    new_model: str = "Llama-2-7b-chat-finetune"
    
@dataclass
class DataConfig:
    """Configuration for data parameters"""
    dataset_name: str = "mlabonne/guanaco-llama2-1k"
    max_seq_length: int = 128
    
@dataclass
class QLoRAConfig:
    """Configuration for QLoRA parameters"""
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
@dataclass
class BitsAndBytesConfig:
    """Configuration for quantization parameters"""
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False
    
@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    output_dir: str = "./results"
    num_train_epochs: int = 1
    fp16: bool = False
    bf16: bool = False
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    max_grad_norm: float = 0.3
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    optim: str = "paged_adamw_32bit"
    lr_scheduler_type: str = "cosine"
    max_steps: int = -1
    warmup_ratio: float = 0.03
    group_by_length: bool = True
    save_steps: int = 0
    logging_steps: int = 25
    packing: bool = False
    device_map: dict = None
    
    def __post_init__(self):
        if self.device_map is None:
            self.device_map = {"": 0}