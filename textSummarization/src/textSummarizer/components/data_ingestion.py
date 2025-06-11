import os
import urllib.request as request
import zipfile
from textSummarizer.utils.common import get_size
from textSummarizer.logging import logger
from textSummarizer.entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        
    def download_file(self):
        if not os.path.exists(self.config.local_data_file) or get_size(self.config.local_data_file) == 0:
            filename,header=request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )
            logger.info(f"Downloaded file: {filename} with headers: {header}")
        else:
            logger.info(f"File already exists of size {get_size(self.config.local_data_file)}")
    def extract_zip_file(self):
        unzip_dir = self.config.unzip_dir
        os.makedirs(unzip_dir, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_dir)
        logger.info(f"Extracted zip file to {unzip_dir}")