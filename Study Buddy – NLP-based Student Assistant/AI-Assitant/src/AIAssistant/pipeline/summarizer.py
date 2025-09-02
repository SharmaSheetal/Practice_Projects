import sys
import os
import numpy as np
import json
from collections import defaultdict
from transformers import T5Tokenizer, T5ForConditionalGeneration
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(ROOT_DIR)
from src.AIAssistant.preprocessing.data_preprocessing import data_preprocessing
from src.AIAssistant import logger
# Load the model and tokenizer
model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
# Summarization function
def summarize(text, max_length, min_length=40):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", truncation=True, max_length=512)  # Truncate to 512
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
def text_summarization(docs):
    """
    Summarize the text data using T5 model.
    """
    logger.info("Text Summarization started.")
    grouped_content = defaultdict(lambda: {"contents": []})  
    result_summaries = defaultdict(lambda: {"summary": None, "title": None})
    for doc in docs:
        temp = doc.to_dict()
        grouped_content[temp["meta"]["name"]]["contents"].append(temp["content"])  # Append content to the list

    for doc_name, contents in grouped_content.items():
        doc_summary = []
        doc_title = None
        contents = contents["contents"]
        for i, content in enumerate(contents):
            
            logger.info(f"Summarizing content for {doc_name} chunk {i+1}")
            doc_summary.append(summarize(content, len(content)//2))
        doc_title = summarize(" ".join(doc_summary),10,5)
        result_summaries[doc_name]["summary"] = " ".join(doc_summary)
        result_summaries[doc_name]["title"] = doc_title
    logger.info("Text Summarization completed.")
    return result_summaries

if __name__ == "__main__":
    doc_summary=text_summarization()
    print(doc_summary)
    


