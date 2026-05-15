import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as T
import os

def run_onnx_inference(onnx_path, image_path):
    """
    Runs inference using an exported ONNX model.
    """
    print(f"--- Running Inference ---")
    print(f"Model: {onnx_path}")
    print(f"Image: {image_path}")

    # 1. Check if files exist
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX model not found at '{onnx_path}'")
        return
    if not os.path.exists(image_path):
        print(f"Error: Image not found at '{image_path}'")
        return

    # 2. Load the ONNX model
    # We use CPUExecutionProvider for standard Kaggle environments. 
    # If GPU is available, you can add 'CUDAExecutionProvider' to the list.
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    # 3. Define image preprocessing 
    # CRITICAL: This must exactly match the 192x192 size and ImageNet normalization we used in training!
    preprocess = T.Compose([
        T.Resize((192, 192)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Load and preprocess the test image
    image = Image.open(image_path).convert('RGB')
    
    # Add batch dimension and convert PyTorch Tensor to Numpy Array (ONNX requirement)
    input_tensor = preprocess(image).unsqueeze(0).numpy()  

    # 5. Run Inference!
    # 'input' and 'output' are the exact layer names we defined during the ONNX export in demon_shave_trainer.py
    outputs = session.run(['output'], {'input': input_tensor})
    logits = outputs[0]

    # 6. Get the predicted class
    predicted_class = np.argmax(logits, axis=1)[0]
    
    # You can also get confidence scores using softmax
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)
    
    probabilities = softmax(logits)[0]
    confidence = probabilities[predicted_class]

    print(f"\nPrediction Results:")
    print(f"Class Index: {predicted_class}")
    print(f"Confidence:  {confidence:.2%}")
    print(f"Raw Logits:  {logits[0]}")
    
    return predicted_class, confidence

if __name__ == "__main__":
    # Example usage:
    # Change these paths to test on your own data
    MODEL_PATH = "student_BETA-LION_SHAVED.onnx" 
    
    # For testing purposes, you can point this to an actual image in your dataset
    TEST_IMAGE_PATH = "../competition_dataset/NEU-DET_open/validation/images/crazing/crazing_241.jpg" 
    
    run_onnx_inference(MODEL_PATH, TEST_IMAGE_PATH)
