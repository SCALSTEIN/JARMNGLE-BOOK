# Import necessary libraries
import time
import picamera
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

# Initialize TensorFlow Lite interpreter for image classification
interpreter = tflite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Define model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess image for model input
def preprocess_image(image_path, input_shape):
    image = Image.open(image_path).resize(input_shape)
    image = np.asarray(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image.astype(input_details[0]['dtype'])

# Function to classify an image using the loaded TensorFlow Lite model
def classify_image(image):
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

# Function to capture image from camera and classify
def capture_and_classify():
    with picamera.PiCamera() as camera:
        camera.resolution = (640, 480)
        camera.start_preview()
        time.sleep(2)  # Allow camera to warm up

        # Capture image to a numpy array
        image = np.empty((480, 640, 3), dtype=np.uint8)
        camera.capture(image, 'rgb')

        # Preprocess and classify image
        preprocessed_image = preprocess_image(image, (224, 224))
        predictions = classify_image(preprocessed_image)

        # Display results (example: assuming binary classification)
        if predictions[0][0] >= 0.5:
            print("Wildlife detected!")
        else:
            print("No wildlife detected.")

# Main function to continuously monitor wildlife
def main():
    while True:
        capture_and_classify()
        time.sleep(10)  # Wait for 10 seconds between each capture

if __name__ == "__main__":
    main()
