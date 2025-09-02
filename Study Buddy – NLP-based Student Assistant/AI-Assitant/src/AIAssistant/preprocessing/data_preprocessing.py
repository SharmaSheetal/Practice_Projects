import sys
import os

# Get the root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(ROOT_DIR)
from src.AIAssistant import logger
from haystack.utils import fetch_archive_from_http
from haystack.nodes import TextConverter, PDFToTextConverter, DocxToTextConverter, PreProcessor
from haystack.utils import convert_files_to_docs
from haystack.nodes import PreProcessor
from pathlib import Path
def data_preprocessing():
    logger.info("Data Preprocessing started.")
    data_dir = Path("artifacts/raw_data")
    files = [file for file in data_dir.iterdir() if file.is_file()]
    logger.info(f"Found {files} in the data directory.")
    logger.info("Converting files to haystack docs.")
    all_docs = convert_files_to_docs(dir_path=data_dir)
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=False,
        split_by="word",
        split_length=500,
        split_respect_sentence_boundary=True,
    )
    docs = preprocessor.process(all_docs)
    logger.info("Data Preprocessing completed.")
    return docs
if __name__ == "__main__":
    docs = data_preprocessing()
    print(docs)
