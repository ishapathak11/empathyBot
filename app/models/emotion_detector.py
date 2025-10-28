# emotion_detector.py

import torch
import re
import os
import json
from transformers import AutoTokenizer, BertForSequenceClassification

class EmotionDetector:
    def __init__(self, use_pretrained=False, model_path='/emotion_classifier_model', save_path=None):
        """
        Initialize emotion detection model with expanded emotion categories

        Args:
            use_pretrained: If True, load from Hugging Face; if False, load from local path
            model_path: Path to saved model if use_pretrained is False
            save_path: If provided, save the model to this path after loading
        """
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Define our expanded emotion labels
        self.emotion_labels = [
            'anger', 'fear', 'sadness', 'joy', 'love', 
            'gratitude', 'neutral', 'curiosity', 'disappointment', 
            'surprise', 'pride'
        ]

        # Define mapping to expanded emotion categories
        self.mapping = {
            # 1. Anger
            'anger':         ('anger',         0.45),
            'disgust':       ('anger',         0.45),
            'annoyance':     ('anger',         0.45),
            'disapproval':   ('anger',         0.45),

            # 2. Fear
            'fear':          ('fear',          0.72),
            'nervousness':   ('fear',          0.72),
            'confusion':     ('fear',          0.72),
            'embarrassment': ('fear',          0.72),

            # 3. Sadness
            'sadness':       ('sadness',       0.90),
            'grief':         ('sadness',       0.90),
            'remorse':       ('sadness',       0.90),

            # 4. Joy
            'joy':           ('joy',           0.29),
            'amusement':     ('joy',           0.29),
            'excitement':    ('joy',           0.29),
            'optimism':      ('joy',           0.29),

            # 5. Love
            'love':          ('love',          0.16),
            'caring':        ('love',          0.16),
            'approval':      ('love',          0.16),
            'desire':        ('love',          0.16),
            'admiration':    ('love',          0.16),

            # 6. Gratitude
            'gratitude':     ('gratitude',     0.62),
            'relief':        ('gratitude',     0.62),

            # 7. Neutral
            'neutral':       ('neutral',       0.4),
            'realization':   ('neutral',       0.4),

            # 8. Curiosity (newly added)
            'curiosity':     ('curiosity',     1.0),

            # 9. Disappointment (newly added)
            'disappointment':('disappointment', 1.38),

            # 10. Surprise (newly added)
            'surprise':      ('surprise',      1.64),

            # 11. Pride (newly added)
            'pride':         ('pride',         3.5)
        }

        # Load model
        if use_pretrained:
            print("Loading pre-trained model from Hugging Face...")
            model_name = "bhadresh-savani/bert-base-go-emotion"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = BertForSequenceClassification.from_pretrained(model_name)
            self.original_labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

            # Save model if requested
            if save_path:
                print(f"Saving model to {save_path}...")
                self._save_model(save_path)
        else:
            if not model_path:
                raise ValueError("model_path must be provided if use_pretrained is False")

            print(f"Loading model from {model_path}...")
            self._load_local_model(model_path)

        self.model.to(self.device).eval()
        print(f"Model loaded successfully and running on {self.device}")

    def _save_model(self, save_path):
        """Save model and tokenizer to disk in a format that can be loaded locally"""
        os.makedirs(save_path, exist_ok=True)

        # Save model
        self.model.save_pretrained(save_path)

        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)

        # Save original labels
        with open(os.path.join(save_path, 'original_labels.json'), 'w') as f:
            json.dump(self.original_labels, f)

        # Save mapping
        with open(os.path.join(save_path, 'emotion_mapping.json'), 'w') as f:
            json.dump(self.mapping, f)

        print(f"Model saved to {save_path}")

    def _load_local_model(self, model_path):
        """Load model from local path, handling various possible structures"""
        # Check if path exists
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")

        try:
            # First try to load with auto classes
            print("Attempting to load with AutoTokenizer and AutoModel...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self.model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)

            # Get original labels
            if os.path.exists(os.path.join(model_path, 'original_labels.json')):
                with open(os.path.join(model_path, 'original_labels.json'), 'r') as f:
                    self.original_labels = json.load(f)
            else:
                self.original_labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

            print("Model loaded successfully with auto classes")

        except Exception as e:
            print(f"Error loading with auto classes: {e}")
            print("Falling back to direct loading...")

            try:
                # Try to load model directly with PyTorch
                model_file = os.path.join(model_path, 'pytorch_model.bin')
                if not os.path.exists(model_file):
                    raise FileNotFoundError(f"Model file not found: {model_file}")

                # Load config if available
                config_path = os.path.join(model_path, 'config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config_dict = json.load(f)

                    # Check config structure and add model_type if missing
                    if 'model_type' not in config_dict:
                        config_dict['model_type'] = 'bert'
                        with open(config_path, 'w') as f:
                            json.dump(config_dict, f, indent=2)
                        print("Added model_type to config.json")

                    from transformers import BertConfig
                    config = BertConfig.from_dict(config_dict)
                    self.model = BertForSequenceClassification(config)
                else:
                    # Create default config
                    from transformers import BertConfig
                    config = BertConfig.from_pretrained('bert-base-uncased', num_labels=28)
                    self.model = BertForSequenceClassification(config)

                # Load model weights
                state_dict = torch.load(model_file, map_location='cpu')
                self.model.load_state_dict(state_dict, strict=False)

                # Load or create tokenizer
                vocab_file = os.path.join(model_path, 'vocab.txt')
                if os.path.exists(vocab_file):
                    from transformers import BertTokenizer
                    self.tokenizer = BertTokenizer(vocab_file=vocab_file)
                else:
                    from transformers import BertTokenizer
                    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

                # Set original labels
                if hasattr(self.model.config, 'id2label'):
                    self.original_labels = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]
                else:
                    # Fallback to default GoEmotions labels
                    self.original_labels = [
                        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
                        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
                        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
                        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
                        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
                    ]

                print("Model loaded successfully with direct loading")

            except Exception as e:
                print(f"Error with direct loading: {e}")
                print("Falling back to using pre-trained model from Hugging Face...")

                model_name = "bhadresh-savani/bert-base-go-emotion"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = BertForSequenceClassification.from_pretrained(model_name)
                self.original_labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    def predict_original_emotions(self, text):
        """Get probabilities for original GoEmotions labels"""
        # Preprocess
        text = self.preprocess_text(text)

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)

        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs[0]

            # GoEmotions uses sigmoid for multi-label classification
            probs = torch.sigmoid(logits)[0].cpu().tolist()

        # Make sure lengths match
        if len(probs) != len(self.original_labels):
            # Pad or truncate probabilities to match labels
            if len(probs) < len(self.original_labels):
                probs = probs + [0.0] * (len(self.original_labels) - len(probs))
            else:
                probs = probs[:len(self.original_labels)]

        return dict(zip(self.original_labels, probs))

    def predict_emotion(self, text):
        """Detect emotion in text using our expanded emotion categories"""
        # Get original emotion probabilities
        original_emotions = self.predict_original_emotions(text)

        # Map to our expanded categories and combine probabilities
        mapped_emotions = {emotion: 0.0 for emotion in self.emotion_labels}

        for original_emotion, prob in original_emotions.items():
            if original_emotion in self.mapping:
                mapped_emotion, weight = self.mapping[original_emotion]
                mapped_emotions[mapped_emotion] += prob * weight

        # Normalize
        total = sum(mapped_emotions.values())
        if total > 0:
            mapped_emotions = {k: v/total for k, v in mapped_emotions.items()}

        # Find primary emotion
        primary_emotion = max(mapped_emotions.items(), key=lambda x: x[1])

        return {
            'emotion': primary_emotion[0],
            'confidence': primary_emotion[1],
            'all_emotions': mapped_emotions,
            'original_emotions': original_emotions
        }

    def preprocess_text(self, text):
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text