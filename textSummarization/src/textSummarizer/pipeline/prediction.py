from textSummarizer.config.configuration import ConfigurationManager
from transformers import pipeline,AutoTokenizer

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()
    
    def predict(self, text):
        """
        Predicts the summary of the given text using a pre-trained model.
        
        Args:
            text (str): The input text to summarize.
        
        Returns:
            str: The predicted summary of the input text.
        """
        gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 128}

        summarization_pipeline = pipeline(
            task="summarization",
            model=self.config.model_ckpt,
            tokenizer=AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        )
        
        summary = summarization_pipeline(text,**gen_kwargs)[0]["summary_text"]
        print(f"Generated Summary: {summary}")
        return summary