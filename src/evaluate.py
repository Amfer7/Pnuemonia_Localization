import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.metrics import classification_report, confusion_matrix
from src.config import *

def generate_cam(model, img_array, class_index):
    """Generates the Class Activation Map (CAM)."""
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=(model.get_layer('last_conv_layer').output, model.output)
    )
    with tf.GradientTape() as tape:
        last_conv_output, preds = grad_model(img_array)
        class_channel = preds[:, class_index]
    
    grads = tape.gradient(class_channel, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_output = last_conv_output[0]
    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = cv2.resize(heatmap.numpy(), (IMG_SIZE, IMG_SIZE))
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap

def main():
    model = load_model('saved_models/best_model.h5')
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        TEST_PATH, target_size=(IMG_SIZE, IMG_SIZE), batch_size=1, class_mode='categorical', shuffle=False)
    
    Y_pred = model.predict(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    
    print('Classification Report:\n', classification_report(test_generator.classes, y_pred, target_names=['Healthy', 'Pneumonia']))

    # Visualize CAM on a test image
    pneumonia_img_path = os.path.join(TEST_PATH, 'Pneumonia', os.listdir(os.path.join(TEST_PATH, 'Pneumonia'))[0])
    original_img = cv2.imread(pneumonia_img_path)
    img_array = np.expand_dims(cv2.resize(original_img, (IMG_SIZE, IMG_SIZE)), axis=0) / 255.0
    
    heatmap = generate_cam(model, img_array, class_index=1) # Class 1 is Pneumonia
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(cv2.resize(original_img, (IMG_SIZE, IMG_SIZE)), 0.6, heatmap, 0.4, 0)

    plt.figure(figsize=(8, 4)); plt.subplot(1, 2, 1); plt.imshow(cv2.cvtColor(cv2.resize(original_img, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB)); plt.title('Original'); plt.subplot(1, 2, 2); plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)); plt.title('CAM Overlay'); plt.show()

if __name__ == '__main__':
    main()