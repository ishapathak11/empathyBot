# response_generator.py

import torch
import random
import json
import os
from transformers import BartTokenizer, BartForConditionalGeneration

class ResponseGenerator:
    def __init__(self, emotion_detector, model_path):
        """
        Initialize the response generator with a fine-tuned BART model

        Args:
            emotion_detector: An instance of EmotionDetector
            model_path: Path to the fine-tuned BART model
        """
        self.emotion_detector = emotion_detector
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Ensure model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        # Load the fine-tuned model
        try:
            print(f"Loading fine-tuned model from {model_path}...")
            self.tokenizer = BartTokenizer.from_pretrained(model_path)
            self.model = BartForConditionalGeneration.from_pretrained(model_path)
            self.model.to(self.device)
            print("Model loaded successfully!")

            # Load emotion labels if available
            emotion_labels_path = os.path.join(model_path, "emotion_labels.json")
            if os.path.exists(emotion_labels_path):
                with open(emotion_labels_path, 'r') as f:
                    self.emotion_labels = json.load(f)
                print(f"Loaded emotion labels: {self.emotion_labels}")
            else:
                self.emotion_labels = emotion_detector.emotion_labels
                print(f"Using emotion detector labels: {self.emotion_labels}")

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to default model...")
            # Fall back to base model
            self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
            self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
            self.model.to(self.device)
            self.emotion_labels = emotion_detector.emotion_labels

    def generate_response(self, message, emotion=None):
        """
        Generate a response for the given message

        Args:
            message: The user's message
            emotion: Optional emotion override (if None, will detect from message)

        Returns:
            dict: The response with context, detected emotion, and generated text
        """
        # Detect emotion if not provided
        if emotion is None:
            result = self.emotion_detector.predict_emotion(message)
            emotion = result['emotion']
            confidence = result['confidence']
            print(f"Detected emotion: {emotion} (confidence: {confidence:.4f})")

        # Format input with emotion
        input_text = f"Emotion: {emotion}. Message: {message}"

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)

        # Generate response
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs["input_ids"],
                max_length=128,
                num_beams=5,
                no_repeat_ngram_size=2,
                top_k=50,
                do_sample=True,  # Enable sampling
                top_p=0.95,
                temperature=0.7
            )

        # Decode response
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return {
            "context": message,
            "detected_emotion": emotion,
            "response": response
        }

