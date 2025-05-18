"""
NLP management module for analyzing financial text using spaCy and OpenAI.
"""

import spacy
import openai
from config import OPENAI_API_KEY

# Initialize spaCy NLP pipeline
nlp = spacy.load("en_core_web_sm")

# Initialize OpenAI API
openai.api_key = OPENAI_API_KEY

class NLPManager:
    @staticmethod
    def analyze_financial_text(text):
        """
        Analyze financial text using spaCy for named entity recognition (NER) 
        and ChatGPT API for sentiment analysis.

        Args:
            text (str): Financial news or text to analyze.

        Returns:
            dict: A dictionary containing named entities and sentiment analysis.
        """
        # Named Entity Recognition with spaCy
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        # Sentiment Analysis with OpenAI API
        try:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=f"Analyze the sentiment of the following financial news:\n\n{text}\n\nSentiment:",
                max_tokens=50,
            )
            sentiment = response['choices'][0]['text'].strip()
        except Exception as e:
            sentiment = f"Error during sentiment analysis: {e}"

        # Combine results
        analysis = {
            "Named Entities": entities,
            "Sentiment Analysis": sentiment
        }
        return analysis 