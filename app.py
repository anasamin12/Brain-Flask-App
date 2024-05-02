from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model loaded successfully.")
print("Input details:", input_details)
print("Output details:", output_details)

@app.route('/classify', methods=['POST'])
def classify_image():
    return jsonify({'message': 'Endpoint is working'})
    image_file = request.files['image']
    if not image_file:
        return jsonify({'error': 'No image file provided.'}), 400

    image_np = np.frombuffer(image_file.read(), np.uint8)
    if image_np.size == 0:
        return jsonify({'error': 'The image data is empty.'}), 400

    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({'error': 'Image decoding failed.'}), 400
    if __name__ == '__main__':
    app.run(debug=True)
  
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
    image = cv2.resize(image, (150, 150))
    image_processed = np.array(image, dtype=np.float32)  
    image_processed = image_processed.reshape(1, 150, 150, 3) 

    print("Preprocessed image shape:", image_processed.shape)
    print("Data type:", image_processed.dtype)

    interpreter.set_tensor(input_details[0]['index'], image_processed)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output_data)

    labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
    return jsonify({'class': labels[predicted_index]})

if __name__ == '__main__':
    app.run(debug=True)
