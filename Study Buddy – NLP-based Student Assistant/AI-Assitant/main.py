import sys
import os
import json
import numpy as np
import atexit
from collections import defaultdict
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from flask_cors import CORS
import requests  # Import the requests library

# Get the root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(ROOT_DIR)

# Import the function from your pipeline
from src.AIAssistant.pipeline.mcq_generation import generate_mcq

app = Flask(__name__)
CORS(app)

# Define the file paths
FILE_PATH = os.path.abspath("artifacts/processed_data/generated_questions_answers.json")
UPLOAD_FOLDER = os.path.abspath("artifacts/raw_data/")  # Folder to store uploaded files
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx'}  # Allowed file types

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_serializable(obj):
    """
    Helper function to convert non-serializable objects (e.g., numpy arrays, sets) into JSON-friendly types.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy arrays to lists
    elif isinstance(obj, set):
        return list(obj)  # Convert sets to lists
    elif isinstance(obj, defaultdict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}  # Recursively convert defaultdict
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}  # Recursively convert dict
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]  # Convert lists
    return obj  # Return other data types as-is

@app.route('/result', methods=['GET'])
def get_mcq():
    """
    API endpoint to generate MCQs using the predefined pipeline.
    """
    try:
        if not os.path.exists(FILE_PATH):
            generate_mcq()  # Call the actual function to generate questions if file doesn't exist

        with open(FILE_PATH, "r") as f:
            mcq_output = json.load(f)

        # Convert non-serializable objects (like NumPy arrays or defaultdicts) to serializable types
        mcq_output_serialized = convert_to_serializable(mcq_output)

        return jsonify(mcq_output_serialized)  # Return the JSON response with properly serialized data

    except Exception as e:
        app.logger.error(f"Error generating MCQs: {str(e)}")
        return jsonify({"error": str(e)}), 500  # Return error if something goes wrong

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    API endpoint to upload a file (pdf, txt, or docx) and save it to the artifacts/raw_data directory.
    After uploading, call the '/result' API to get MCQs.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)  # Secure the filename to prevent directory traversal
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(file_path)  # Save the file to the specified directory
            # Once the file is uploaded, call the '/result' API
            result_response = requests.get("http://localhost:5002/result")

            # If the '/result' API is successful, you can return its response or handle it
            if result_response.status_code == 200:
                result_data = result_response.json()  # Get the JSON data returned by the result API
                return jsonify({"message": "File successfully uploaded", "file_path": file_path, "mcq_data": result_data}), 200
            else:
                return jsonify({"error": "Failed to fetch MCQs from result API", "result_error": result_response.text}), 500

        except Exception as e:
            app.logger.error(f"Failed to save file: {str(e)}")
            return jsonify({"error": f"Failed to save file: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file format"}), 400

def cleanup():
    """Delete the generated file when the session ends."""
    if os.path.exists(FILE_PATH):
        os.remove(FILE_PATH)
        print(f"Deleted {FILE_PATH} on session exit.")

# Register cleanup function to run when Flask shuts down
atexit.register(cleanup)

if __name__ == '__main__':
    app.run(host='localhost', port=5002, debug=True)
