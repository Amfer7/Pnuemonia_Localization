import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from src.model import build_pneumonia_cnn
from src.config import *

def main():
    model = build_pneumonia_cnn()
    
    # The paper found an Adam optimizer with a learning rate of 0.0001 to be most effective [cite: 76, 157]
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # The paper normalizes pixels[cite: 54]; rescale=1./255 is a standard way to achieve this.
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_PATH, target_size=(IMG_SIZE, IMG_SIZE), batch_size=32, class_mode='categorical')
    validation_generator = val_datagen.flow_from_directory(
        VALIDATION_PATH, target_size=(IMG_SIZE, IMG_SIZE), batch_size=32, class_mode='categorical')
    
    os.makedirs('saved_models', exist_ok=True)
    checkpoint = ModelCheckpoint('saved_models/best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    # The model was trained for 20 epochs [cite: 76]
    model.fit(
        train_generator,
        epochs=20,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stopping])
    print("Training finished.")

if __name__ == '__main__':
    main()