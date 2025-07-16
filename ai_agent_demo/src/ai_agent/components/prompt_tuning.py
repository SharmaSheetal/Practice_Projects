from transformers import pipeline
from src.ai_agent.config.configuration import ModelConfig, TrainingConfig

# Load your model (example)
# model_config = ModelConfig()
# training_config = TrainingConfig()
# model_name = "NousResearch/Llama-2-7b-chat-hf"
# pipe = pipeline("text-generation", model=model_name)

# User prompt and system prompt
user_prompt = "Explain the difference between supervised and unsupervised learning."
system_prompt = """You are a helpful and knowledgeable assistant.

Below are examples of how you should respond to user questions:
example 1:

user: Can you write a short introduction about the relevance of the term "monopsony" in economics? Please use examples related to potential monopsonies in the labour market and cite relevant research.
Response: "Monopsony" refers to a market structure where there is only one buyer for a particular good or service. In economics, this term is particularly relevant in the labor market, where a monopsony employer has significant power over the wages and working conditions of their employees. The presence of a monopsony can result in lower wages and reduced employment opportunities for workers, as the employer has little incentive to increase wages or provide better working conditions.
example 2:
user: How would I create a script for Blender that replaces all of objects with a specific name with a point light?
Response: Here's an example Python script for Blender that replaces all objects with a specific name (in this case, "Cube") with a point light:
"""
def generate_response(pipe,query):
    """Generate a response using the model pipeline"""
    # Format the LLaMA 2 chat prompt
    prompt = (
        f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
        f"{query}"
    )

    formatted_prompt = f"<s>[INST] {prompt.strip()} [/INST]"

    # Generate result
    result = pipe(formatted_prompt, max_new_tokens=256, do_sample=True)
    response = result[0]['generated_text']
    return response.split("[/INST]")[-1].strip()
