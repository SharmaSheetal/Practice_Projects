# Agentic AI Demo

This project demonstrates agentic AI techniques using Llama-2 models and QLoRA for efficient fine-tuning and prompt tuning.

## Features

- **Dataset:** `mlabonne/guanaco-llama2-1k`
- **Base Model:** `NousResearch/Llama-2-7b-chat-hf`
- **Fine-tuned Model:** `Llama-2-7b-chat-finetune`
- **QLoRA Config:**
    - `lora_r`: 64
    - `lora_alpha`: 16
    - `lora_dropout`: 0.1
- **BitsAndBytes Config:**
    - `use_4bit`: True
    - `bnb_4bit_compute_dtype`: "float16"
    - `bnb_4bit_quant_type`: "nf4"
    - `use_nested_quant`: False

## Prompt Tuning

- **User Prompt Example:**  
    `Explain the difference between supervised and unsupervised learning.`

- **System Prompt Example:**  
    You are a helpful and knowledgeable assistant.  
    *(See code for detailed examples.)*

## Getting Started

1. **Clone the repository:**
     ```bash
     git clone https://github.com/SharmaSheetal/Practice_Projects.git
     ```
2. **Navigate to the project folder:**
     ```bash
     cd Practice_Projects/ai_agent_demo
     ```
3. **Run the demo:**
     ```bash
     python main.py
     ```

## Example Run

```
Choose a method: 1. fine-tune model, 2. prompt tuning
1
Model path: Llama-2-7b-chat-finetune
Model path exists: True
Loading base model from: NousResearch/Llama-2-7b-chat-hf
Loading LoRA adapter from: ./results
Loading checkpoint shards: 100%|█| 2/2 [01:08<00:00, 34.08s/it]
Model and pipeline loaded successfully!

==================================================
CHAT READY - Model loaded and cached!
==================================================
Enter your query (or type 'exit' to quit): explain rag systems
==================================================
GENERATING RESPONSE...
==================================================
Pipeline not found, creating new one...
Query: explain rag systems
------------------------------
Response: Rag systems are a way of organizing and managing data in a computer system.

In a rag system, data is organized into separate files called "rags". Each rags contains a subset of the data that is being managed by the system, and the system can access and manipulate the data in each rags independently.

Rag systems are often used in distributed systems where data is spread across multiple machines, or in systems where the data is changing rapidly and it is not practical to store all of the data in a single location. By using rags, the system can efficiently manage large amounts of data by breaking it up into smaller, more manageable pieces.
```

Results are saved to `result.json`.

## License

See [LICENSE](../LICENSE) for details.