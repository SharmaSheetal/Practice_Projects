from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
import torch
from transformers import pipeline, logging
from src.ai_agent.config.configuration import ModelConfig, TrainingConfig
from src.ai_agent.components.prompt_tuning import generate_response

# Updated DuckDuckGo search import
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    search = DuckDuckGoSearchRun()
except ImportError:
    # Fallback for newer versions
    try:
        from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
        from langchain_community.tools import DuckDuckGoSearchResults
        search_wrapper = DuckDuckGoSearchAPIWrapper()
        search = DuckDuckGoSearchResults(api_wrapper=search_wrapper)
    except ImportError:
        # Manual implementation if all else fails
        def manual_search(query):
            try:
                import ddgs
                results = ddgs.DDGS().text(query, max_results=3)
                return "\n".join([f"{r['title']}: {r['body']}" for r in results])
            except:
                return f"Unable to search for: {query}"
        
        class ManualSearchTool:
            def run(self, query):
                return manual_search(query)
        
        search = ManualSearchTool()

def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {filename}"

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file.",
)

search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, logging
from peft import PeftModel
import os
from src.ai_agent.config.configuration import ModelConfig, TrainingConfig

class FineTuneModelTool:
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        self.model_config = model_config
        self.training_config = training_config
        self.model = None
        self.tokenizer = None
        self.pipeline = None

    def load_model(self):
        """Load model with LoRA adapter support"""
        if self.model is not None and self.tokenizer is not None:
            print("Model already loaded, skipping reload...")
            return self.model, self.tokenizer
            
        try:
            logging.set_verbosity(logging.ERROR)
            
            # Check if we have a complete model or just LoRA adapters
            adapter_path = "./results"  # Your LoRA adapter path
            base_model_path = self.model_config.model_name  # Original model
            complete_model_path = "./Llama-2-7b-chat-finetune"
            
            # Try to load complete model first
            if os.path.exists(complete_model_path) and os.path.exists(os.path.join(complete_model_path, "config.json")):
                print(f"Loading complete fine-tuned model from: {complete_model_path}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    complete_model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained(complete_model_path)
                
            # If complete model doesn't exist, load base model + LoRA adapter
            elif os.path.exists(adapter_path) and os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
                print(f"Loading base model from: {base_model_path}")
                print(f"Loading LoRA adapter from: {adapter_path}")
                
                # Load base model
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True
                )
                
                # Load LoRA adapter
                self.model = PeftModel.from_pretrained(base_model, adapter_path)
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
                
            else:
                # Fallback to base model only
                print(f"No fine-tuned model found, loading base model: {base_model_path}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True
                )
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            
            # Setup tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            print("Model and pipeline loaded successfully!")
            return self.model, self.tokenizer
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def run_model(self, model, tokenizer, query):
        """Generate response using the model"""
        print("=" * 50)
        print("GENERATING RESPONSE...")
        print("=" * 50)
        
        try:
            if self.pipeline is None:
                print("Pipeline not found, creating new one...")
                self.pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    return_full_text=False
                )
            
            print(f"Query: {query}")
            print("-" * 30)
            
            # Format prompt for Llama-2-chat
            formatted_prompt = f"<s>[INST] {query} [/INST]"
            
            # Generate response
            result = self.pipeline(formatted_prompt)
            
            if result and len(result) > 0:
                response = result[0]['generated_text'].strip()
                print(f"Response: {response}")
                return [{
                    "prompt": query,
                    "response": response
                }]
            else:
                print("No response generated")
                return [{"prompt": query, "error": "No response generated"}]
                
        except Exception as e:
            print(f"Error generating response: {e}")
            return [{"prompt": query, "error": str(e)}]


chat_bot = FineTuneModelTool(    model_config=ModelConfig(),
    training_config=TrainingConfig()
)
chattool = Tool(
    name="chat_bot",
    func=chat_bot.run_model,
    description="Chat with the fine-tuned model using a query",
)
def generate_response_promt(pipe,qyery):
    return generate_response(pipe,qyery)

    

chattool_prompt = Tool(
    name="chat_bot_prompt",
    func=generate_response_promt,
    description="Chat with the fine-tuned model using a prompt",
)