from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

# Load and prepare the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print model details to the console (optional, useful for debugging)
print("Model loaded successfully.")
print("Input details:", input_details)
print("Output details:", output_details)

@app.route('/classify', methods=['POST'])
def classify_image():
    # Get the image file from the request
    image_file = request.files['image']
    if not image_file:
        return jsonify({'error': 'No image file provided.'}), 400

    # Convert the image file to a NumPy array
    image_np = np.frombuffer(image_file.read(), np.uint8)
    if image_np.size == 0:
        return jsonify({'error': 'The image data is empty.'}), 400

    # Decode the image
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({'error': 'Image decoding failed.'}), 400

    # Preprocess the image for the model
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (150, 150))
    image_processed = np.array(image, dtype=np.float32)
    image_processed = image_processed.reshape(1, 150, 150, 3)

    # Print preprocessed image details (optional, useful for debugging)
    print("Preprocessed image shape:", image_processed.shape)
    print("Data type:", image_processed.dtype)

    # Run the TensorFlow Lite model
    interpreter.set_tensor(input_details[0]['index'], image_processed)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output_data)

    # Define the class labels and return the result
    labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
    return jsonify({'class': labels[predicted_index]})

# Do not include app.run() to make it compatible with Vercel
