from src.ai_agent.components.model_evalaution import ModelEvaluator
from src.ai_agent.config.configuration import ModelConfig, TrainingConfig

class EvaluationPipeline:
    def __init__(self):
        self.model_config = ModelConfig()
        self.training_config = TrainingConfig()
        self.evaluator = ModelEvaluator(self.model_config, self.training_config)
        
    def run_evaluation_pipeline(self, model, tokenizer, test_prompts=None):
        """Run the evaluation pipeline"""
        print("=" * 80)
        print("STARTING EVALUATION PIPELINE")
        print("=" * 80)
        
        # Evaluate model
        results = self.evaluator.evaluate_model(model, tokenizer, test_prompts)
        
        # Save results
        self.evaluator.save_evaluation_results(results)
        
        print("=" * 80)
        print("EVALUATION PIPELINE COMPLETED")
        print("=" * 80)
        
        return results