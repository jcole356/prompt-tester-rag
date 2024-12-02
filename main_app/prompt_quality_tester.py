# prompt_quality_tester.py
import spacy
import numpy as np
from langchain.llms import OpenAI
from textblob import TextBlob
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict
from rag_service_client import RAGServiceClient  # Import RAGServiceClient for integration
from textstat import flesch_reading_ease
from sklearn.metrics.pairwise import cosine_similarity

class PromptQualityTester:
    def __init__(self, api_key: str, rag_client: RAGServiceClient):
        """Initialize the tester with OpenAI API key and RAG client"""
        self.llm = OpenAI(api_key=api_key)
        self.nlp = spacy.load('en_core_web_sm')
        self.history = []
        self.rag_client = rag_client

    def test_prompt(self, prompt: str, expected_pattern: str) -> Dict:
        """Test a prompt and return quality metrics.

        Args:
            prompt (str): The prompt to be tested (either basic or enhanced).
            expected_pattern (str): The expected structure or content for comparison.
            is_enhanced (bool): Flag to indicate if the prompt was enhanced via RAG pipeline.

        Returns:
            Dict: A dictionary containing metrics, prompt, response, and other details.
        """
        try:
            # Generate response from LLM
            response = self.llm.generate([prompt]).generations[0][0].text
            
            # Calculate metrics
            metrics = self.evaluate_response(prompt, expected_pattern, response)
            
            # Save result
            result = {
                'timestamp': datetime.now().isoformat(),
                'prompt': prompt,
                'expected': expected_pattern,
                'response': response,
                'metrics': metrics
            }
            self.history.append(result)
            return result
            
        except Exception as e:
            return {"error": f"Error testing prompt: {str(e)}"}
        
    def evaluate_response(self, prompt: str, expected: str, actual: str) -> Dict:
        """Calculate quality metrics for the response"""
        metrics = {
            'clarity': self._measure_clarity(actual),
            'relevance': self._measure_relevance(expected, actual),
            'completeness': self._measure_completeness(expected, actual),
            'consistency': self._measure_consistency(actual),
            'conciseness': self._measure_conciseness(actual)
        }
        metrics['overall'] = np.mean(list(metrics.values()))
        return metrics

    def _measure_clarity(self, text: str) -> float:
        """Measure text clarity using readability metrics"""
        try:
            # Using Flesch Reading Ease score (higher score = easier to read)
            readability_score = flesch_reading_ease(text)
            # Normalize to a 0-1 scale (adjust as needed based on readability range)
            return max(0.0, min(1.0, readability_score / 100))
        except Exception as e:
            print(f"Error calculating clarity: {e}")
            return 0.0

    def _measure_relevance(self, expected: str, actual: str) -> float:
        """Measure semantic similarity between expected and actual using embeddings"""
        try:
            expected_embedding = self.rag_client.get_embedding(expected)  # Assuming embeddings can be fetched like this
            actual_embedding = self.rag_client.get_embedding(actual)
            similarity_score = cosine_similarity([expected_embedding], [actual_embedding])[0][0]
            return similarity_score
        except Exception as e:
            print(f"Error calculating relevance: {e}")
            return 0.0

    def _measure_completeness(self, expected: str, actual: str) -> float:
        """Measure if all expected elements are present"""
        expected_chunks = set(chunk.text.lower() for chunk in self.nlp(expected).noun_chunks)
        actual_chunks = set(chunk.text.lower() for chunk in self.nlp(actual).noun_chunks)
        
        expected_entities = set(ent.text.lower() for ent in self.nlp(expected).ents)
        actual_entities = set(ent.text.lower() for ent in self.nlp(actual).ents)
        
        expected_keys = expected_chunks.union(expected_entities)
        actual_keys = actual_chunks.union(actual_entities)
        
        if len(expected_keys) == 0:
            return 1.0
        return len(actual_keys.intersection(expected_keys)) / len(expected_keys)


    def _measure_consistency(self, text: str) -> float:
        """Measure internal consistency of response"""
        doc = self.nlp(text)
        sentences = list(doc.sents)
        if len(sentences) <= 1:
            return 1.0
        similarities = []
        for i in range(len(sentences)-1):
            similarities.append(sentences[i].similarity(sentences[i+1]))
        return np.mean(similarities)

    def _measure_conciseness(self, text: str) -> float:
        """Measure text conciseness"""
        words = len(text.split())
        return min(1.0, 2.0 / (1 + np.exp(words / 100)))

def create_radar_chart(metrics: dict):
    """Create a radar chart of metrics"""
    # Remove overall score from radar chart
    display_metrics = {k: v for k, v in metrics.items() if k != 'overall'}
    
    fig = go.Figure(data=go.Scatterpolar(
        r=list(display_metrics.values()),
        theta=list(display_metrics.keys()),
        fill='toself'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False
    )
    return fig