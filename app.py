import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# --- Re-using the CAM logic from evaluate.py ---
def generate_grad_cam(model, img_array, class_index):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=(model.get_layer('last_conv_layer').output, model.output)
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs = conv_outputs[0]

    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)

    # Normalize and resize the heatmap
    cam = cv2.resize(cam.numpy(), (128, 128))
    cam = np.maximum(cam, 0)
    if np.max(cam) > 0:
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    return cam

def get_bounding_box(heatmap):
    """Simplified bounding box generation from the heatmap[cite: 138]."""
    _, thresh = cv2.threshold(np.uint8(255 * heatmap), 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return cv2.boundingRect(max(contours, key=cv2.contourArea))
    return None

# --- Flask App Setup ---
app = Flask(__name__)
model = load_model('saved_models/best_model.h5')

@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join('static/uploads', file.filename)
            file.save(filepath)
            
            img = cv2.imread(filepath)
            img_resized = cv2.resize(img, (128, 128))
            img_array = np.expand_dims(img_resized, axis=0) / 255.0

            prediction = model.predict(img_array)[0]
            class_index = np.argmax(prediction)
            label = ['Healthy', 'Pneumonia'][class_index]
            confidence = prediction[class_index]
            
            output_img = cv2.resize(img, (512, 512)) # Use a larger image for display
            if label == 'Pneumonia':
                heatmap = generate_grad_cam(model, img_array, 1)
                box = get_bounding_box(heatmap)
                if box:
                    x, y, w, h = [val * 4 for val in box] # Scale box to 512x512
                    cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            cv2.putText(output_img, f'{label} ({confidence:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            output_path = os.path.join('static/predictions', file.filename)
            cv2.imwrite(output_path, output_img)

            return render_template('index.html', label=label, confidence=confidence, image_file=output_path)
    return render_template('index.html', label=None, confidence=None, image_file=None)

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('static/predictions', exist_ok=True)
    app.run(debug=True)