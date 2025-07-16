import torch
from transformers import pipeline, logging
from ai_agent.config.configuration import ModelConfig, TrainingConfig

class ModelEvaluator:
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        self.model_config = model_config
        self.training_config = training_config
        
    def evaluate_model(self, model, tokenizer, test_prompts=None):
        """Evaluate the trained model"""
        # Suppress warnings
        logging.set_verbosity(logging.CRITICAL)
        
        if test_prompts is None:
            test_prompts = [
                "What is a large language model?",
                "Explain machine learning in simple terms",
                "How does fine-tuning work?",
                "What are the benefits of using LoRA?"
            ]
        
        print("=" * 80)
        print("MODEL EVALUATION")
        print("=" * 80)
        
        # Create pipeline
        pipe = pipeline(
            task="text-generation", 
            model=model, 
            tokenizer=tokenizer, 
            max_length=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
        
        results = []
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            print("-" * 40)
            
            # Format prompt for Llama-2-chat
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            
            try:
                result = pipe(formatted_prompt)
                response = result[0]['generated_text']
                
                # Extract only the generated part
                if "[/INST]" in response:
                    response = response.split("[/INST]")[-1].strip()
                
                print(f"Response: {response}")
                
                results.append({
                    "prompt": prompt,
                    "response": response,
                    "full_output": result[0]['generated_text']
                })
                
            except Exception as e:
                print(f"Error generating response: {e}")
                results.append({
                    "prompt": prompt,
                    "error": str(e)
                })
        
        return results
    
    def save_evaluation_results(self, results, filename="evaluation_results.txt"):
        """Save evaluation results to file"""
        with open(filename, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL EVALUATION RESULTS\n")
            f.write("=" * 80 + "\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"Test {i}:\n")
                f.write(f"Prompt: {result['prompt']}\n")
                
                if 'error' in result:
                    f.write(f"Error: {result['error']}\n")
                else:
                    f.write(f"Response: {result['response']}\n")
                
                f.write("\n" + "-" * 40 + "\n\n")
        
        print(f"Evaluation results saved to {filename}")