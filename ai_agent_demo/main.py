from src.ai_agent.pipeline.training_pipeline import TrainingPipeline
from src.ai_agent.pipeline.evaluation_pipeline import EvaluationPipeline
from src.ai_agent.components.tool import FineTuneModelTool, save_tool, search_tool, wiki_tool, chattool, chattool_prompt
from src.ai_agent.config.configuration import ModelConfig, TrainingConfig
import os
import json
from datetime import datetime
from transformers import pipeline, logging

def load_existing_results():
    """Load existing results from result.json or create empty list"""
    try:
        if os.path.exists("result.json"):
            with open("result.json", "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return []
    except (json.JSONDecodeError, FileNotFoundError):
        return []

def save_result(choice_name, query, response):
    """Save query and response to result.json"""
    results = load_existing_results()
    
    new_entry = {
        "timestamp": datetime.now().isoformat(),
        "choice": choice_name,
        "query": query,
        "response": response
    }
    
    results.append(new_entry)
    
    try:
        with open("result.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Result saved to result.json")
    except Exception as e:
        print(f"Error saving to result.json: {e}")

def main():
    """Main function to run the complete pipeline"""
    choice = input("Choose a method: 1. fine-tune model, 2. prompt tuning\n")
    
    if choice == "1":
        choice_name = "fine-tune model"
        model_config = ModelConfig()
        training_config = TrainingConfig()
        
        # if result is not found, then run training pipeline
        if not os.path.exists("./results"):
            # Run training pipeline
            training_pipeline = TrainingPipeline()
            model, tokenizer = training_pipeline.run_training_pipeline()
            
            # Run evaluation pipeline
            evaluation_pipeline = EvaluationPipeline()
            results = evaluation_pipeline.run_evaluation_pipeline(model, tokenizer)
            
            print("Complete pipeline finished successfully!")
        
        # Debug model configuration
        print(f"Model path: {model_config.new_model}")
        print(f"Model path exists: {os.path.exists(model_config.new_model)}")
        
        # Initialize the chat bot and load the model
        chat_bot = FineTuneModelTool(
            model_config=model_config,
            training_config=training_config
        )
        
        try:
            model, tokenizer = chat_bot.load_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            return
        
        # Chat loop
        print("\n" + "="*50)
        print("CHAT READY - Model loaded and cached!")
        print("="*50)
        
        while True:
            query = input("Enter your query (or type 'exit' to quit): ")
            if query.lower() == 'exit':
                break
            
            try:
                response = chattool.func(model, tokenizer, query)
                print(f"Response: {response}")
                
                # Save to result.json
                save_result(choice_name, query, response)
                
            except Exception as e:
                print(f"Error during chat: {e}")
                # Save error to result.json as well
                save_result(choice_name, query, f"Error: {str(e)}")
    
    else:
        choice_name = "prompt tuning"
        model_name = "NousResearch/Llama-2-7b-chat-hf"
        pipe = pipeline("text-generation", model=model_name)
        
        while True:
            query = input("Enter your query (or type 'exit' to quit): ")
            if query.lower() == 'exit':
                break
            
            try:
                response = chattool_prompt.func(pipe, query)
                print(f"Response: {response}")
                
                # Save to result.json
                save_result(choice_name, query, response)
                
            except Exception as e:
                print(f"Error during chat: {e}")
                # Save error to result.json as well
                save_result(choice_name, query, f"Error: {str(e)}")

if __name__ == "__main__":
    main()