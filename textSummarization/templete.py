from pathlib import Path
import logging
import os

logging.basicConfig(level=logging.INFO,format='[%(asctime)s]: %(message)s')

project_name = "textSummarizer"
list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/logging/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    f"config/config.yaml",
    "params.yaml",
    "app.py",
    "requirements.txt",
    "setup.py",
    "main.py",
    "Dockerfile",
    "research/trails.ipynb"]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir !="":
        logging.info(f"Creating directory: {filedir} if it does not exist")
        os.makedirs(filedir, exist_ok=True)
    if not os.path.exists(filepath):
        logging.info(f"Creating file: {filename}")
        with open(filepath, 'w') as f:
            pass
    else:
        logging.info(f"File {filename} already exists. Skipping creation.")
logging.info(f"All files and directories for the project '{project_name}' have been created successfully.")
    