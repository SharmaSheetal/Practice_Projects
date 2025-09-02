import sys
import os
import numpy as np
import json
from collections import defaultdict
# Get the root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(ROOT_DIR)
from src.AIAssistant.pipeline.question_generation import generate_QA
from src.AIAssistant.preprocessing.data_preprocessing import data_preprocessing
from src.AIAssistant import logger
from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines import ExtractiveQAPipeline
from haystack.nodes import BM25Retriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from src.AIAssistant.pipeline.summarizer import text_summarization

docs = data_preprocessing()
document_store = InMemoryDocumentStore(use_bm25=True)
document_store.write_documents(docs)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
retriever = BM25Retriever(document_store=document_store)
pipe = ExtractiveQAPipeline(reader, retriever)
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
def generate_mcq():
    generated_questions = generate_QA(docs)
    summarized_docs = text_summarization(docs)
    mcq_questions = defaultdict(lambda: {
        "queries": [],  
        "id": None,
        "summary": None
    })
    for doc_name, data in generated_questions.items():
        logger.info(f"Document: {doc_name}")
        print(f"Document: {doc_name}")
        for query_,answers in zip(data["queries"], data["answers"]):
            print("*"*50)
            print(f"Query: {query_},type:{type(query_)}")
            prediction = pipe.run(
                query=query_,
                params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}} 
                           )
            # Structure each query as an entry in the list
            query_data = {
                "query": query_,
                "correct answers": answers,
                "generated_options": prediction["answers"],
            }
            mcq_questions[doc_name]['queries'].append(query_data)
            mcq_questions[doc_name]['id']=generated_questions[doc_name]['metadata']['id']
            mcq_questions[doc_name]['summary']=summarized_docs[doc_name]
    with open("artifacts/processed_data/generated_questions_answers.json", "w") as f:
        json.dump(serialize_haystack_object(mcq_questions), f, indent=2)
    logger.info(f"Generated MCQs")
    
if __name__ == "__main__":
   generate_mcq()