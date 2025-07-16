from src.ai_agent.components.data_ingestion import DataIngestion
from src.ai_agent.components.model_trainer import ModelTrainer
from src.ai_agent.components.model_evalaution import ModelEvaluator
from src.ai_agent.config.configuration import ModelConfig, DataConfig, QLoRAConfig, BitsAndBytesConfig, TrainingConfig

class TrainingPipeline:
    def __init__(self):
        # Initialize configurations
        self.model_config = ModelConfig()
        self.data_config = DataConfig()
        self.qlora_config = QLoRAConfig()
        self.bnb_config = BitsAndBytesConfig()
        self.training_config = TrainingConfig()
        
        # Initialize components
        self.model_trainer = ModelTrainer(
            self.model_config, 
            self.qlora_config, 
            self.bnb_config, 
            self.training_config
        )
        
    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        print("=" * 80)
        print("STARTING TRAINING PIPELINE")
        print("=" * 80)
        
        # Step 1: Load model and tokenizer
        model, tokenizer = self.model_trainer.load_model_and_tokenizer()
        
        # Step 2: Setup PEFT
        model = self.model_trainer.setup_peft_model()
        
        # Step 3: Data ingestion
        data_ingestion = DataIngestion(self.data_config, tokenizer)
        train_dataset, data_collator = data_ingestion.prepare_data()
        
        # Step 4: Initialize trainer
        trainer = self.model_trainer.initialize_trainer(train_dataset, data_collator)
        
        # Step 5: Train model
        self.model_trainer.train()
        
        # Step 6: Save model
        self.model_trainer.save_model()
        
        print("=" * 80)
        print("TRAINING PIPELINE COMPLETED")
        print("=" * 80)
        
        return model, tokenizer