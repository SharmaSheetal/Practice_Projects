import sys
import os
import numpy as np
# Get the root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(ROOT_DIR)
from src.AIAssistant import logger
from src.AIAssistant.preprocessing.data_preprocessing import data_preprocessing
from pprint import pprint
from tqdm.auto import tqdm
from haystack.nodes import QuestionGenerator, BM25Retriever, FARMReader
from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines import (
    QuestionAnswerGenerationPipeline,
)
import os
import json


# Initialize Question Generator
logger.info("Initializing Question Generator...")
question_generator = QuestionGenerator()

def serialize_haystack_object(obj):
    """ 
    Convert Haystack Answer, Document, and NumPy objects into JSON-friendly format. 
    Args:
        obj: Haystack object to be serialized
    Returns:
        JSON-friendly object
    
    """
    if isinstance(obj, list):
        return [serialize_haystack_object(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: serialize_haystack_object(value) for key, value in obj.items()}
    elif hasattr(obj, "to_dict"):  # Convert Answer, Document, or other Haystack objects
        return obj.to_dict()
    elif isinstance(obj, np.float32):  # Convert np.float32 to standard float
        return float(obj)
    return obj  # Leave primitives unchanged
def generate_QA(docs):
    """
    Generate questions and answers for each document in the data directory.
    """
    logger.info("Question Generator initialized.")
    logger.info("Initializing Question Answer Generation Pipeline...")
    logger.info("Initializing Question Answer Generation Pipeline...")

    document_store = InMemoryDocumentStore()
    document_store.write_documents(docs)
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

    # Initialize Question Generation Pipeline
    qag_pipeline = QuestionAnswerGenerationPipeline(question_generator, reader)

    logger.info("Question Answer Generation Pipeline initialized.")

    generated_questions={}
    for idx, document in enumerate(tqdm(document_store.get_all_documents())):
        
        result = qag_pipeline.run(documents=[document])  # Full result

        
        # Extract metadata
        metadata = {
            "name": document.meta.get("name", f"doc_{idx}"),
            "content": document.content,
            "id": document.id
        }

        # Store in dictionary
        generated_questions[metadata["name"]] = {
            "metadata": metadata,
            "queries": serialize_haystack_object(result['queries']) ,
            "answers": serialize_haystack_object(result['answers'])
        }

    # Write results to JSON file
    return generated_questions


    # print(prediction)   
            
            

    

# if __name__ == "__main__":
#     generated_questions = generate_QA()
#     with open("artifacts/generated_questions.json", "w") as f:
#         json.dump(generated_questions, f)