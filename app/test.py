# test.py
import os
import sys
from models.emotion_detector import EmotionDetector
from models.response_generator import ResponseGenerator

def test():
    try:
        # Define model paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        emotion_model_path = os.path.join(base_dir,  'models', 'emotion_classifier_model')
        response_model_path = os.path.join(base_dir, 'models', 'emotion_aware_model')
        
        # Print paths for verification
        print(f"Emotion model path: {emotion_model_path}")
        print(f"Response model path: {response_model_path}")
        
        # Check if paths exist
        print(f"Emotion model path exists: {os.path.exists(emotion_model_path)}")
        print(f"Response model path exists: {os.path.exists(response_model_path)}")
        
        # Try loading models
        print("Loading emotion detector...")
        emotion_detector = EmotionDetector(
            use_pretrained=False, 
            model_path=emotion_model_path
        )
        
        print("Loading response generator...")
        response_generator = ResponseGenerator(
            emotion_detector,
            model_path=response_model_path
        )
        
        # Test a prediction
        test_message = "I'm feeling happy today!"
        print(f"Testing prediction with message: '{test_message}'")
        
        result = response_generator.generate_response(test_message)
        print(f"Detected emotion: {result['detected_emotion']}")
        print(f"Response: {result['response']}")
        
        print("All tests passed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()