from fastapi import FastAPI
import uvicorn
import sys
import os
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from textSummarizer.pipeline.prediction import PredictionPipeline

app = FastAPI()

@app.get("/",tags=["authentication"])
async def index():
    """
    Redirects to the /docs endpoint for API documentation.
    """
    return RedirectResponse(url="/docs")

@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response(content="Training started successfully.", media_type="text/plain")
    except Exception as e:
        return Response(content=f"An error occurred while starting training: {str(e)}", media_type="text/plain")
@app.get("/predict")
async def predict(text: str):
    """
    Predicts the summary of the given text using a pre-trained model.
    
    Args:
        text (str): The input text to summarize.
    
    Returns:
        str: The predicted summary of the input text.
    """
    try:
        prediction_pipeline = PredictionPipeline()
        summary = prediction_pipeline.predict(text)
        return Response(content=summary, media_type="text/plain")
    except Exception as e:
        return Response(content=f"An error occurred during prediction: {str(e)}", media_type="text/plain")
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)