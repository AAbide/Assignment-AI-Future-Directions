
# Simple Raspberry Pi Deployment Code
# Save as: raspberry_pi_deploy.py

import tensorflow as tf
import numpy as np
import cv2

# Load the model
interpreter = tf.lite.Interpreter(model_path='recyclable_classifier.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['plastic', 'paper', 'glass', 'metal', 'organic']

def classify_image(image_path):
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Run inference
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    
    # Get result
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output)
    confidence = np.max(output)
    
    return class_names[predicted_class], confidence

# Example usage:
# result, confidence = classify_image('recyclable_item.jpg')
# print(f"This is {result} with {confidence:.2f} confidence")
