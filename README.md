Pneumonia Detection with Explainable AI

This project is an end-to-end deep learning solution for detecting pneumonia from chest X-ray images. It features a custom 10-layer Convolutional Neural Network (CNN) and an interactive web application built with Flask. The application uses Explainable AI (XAI) techniques, specifically Grad-CAM, to generate heatmaps that visually highlight areas of interest in the X-ray, providing transparency for the model's predictions.

🚀 Key Features

    Interactive Web Application: A user-friendly interface built with Flask to upload images and receive instant predictions.

    Explainable AI (XAI): Generates Grad-CAM heatmaps and bounding boxes on predicted pneumonia cases to explain the model's reasoning.

    Custom CNN Architecture: A 10-layer CNN designed for image classification, compatible with CAM visualization techniques.

    Reproducible ML Pipeline: Includes scripts for data preprocessing, model training, and evaluation.

    Balanced Dataset: The preprocessing script balances the dataset to prevent model bias towards a specific class.

📂 Project Structure

.
├── app.py                  # Flask web application for inference
├── config.py               # Configuration settings (paths, image size, etc.)
├── data_preprocessing.py   # Script to process and prepare the dataset
├── evaluate.py             # Script to evaluate the model and generate CAM
├── model.py                # Defines the CNN architecture
├── requirements.txt        # Project dependencies
├── train.py                # Script to train the CNN model
├── raw_data/               # Directory for raw DICOM images and labels (not included)
├── data/                   # Directory for processed PNG images (created by data_preprocessing.py)
├── saved_models/           # Directory for saved model weights (created by train.py)
└── static/                 # Directory for web app assets (uploads, predictions)

🛠️ Setup and Installation

Follow these steps to set up the project environment.

    Clone the repository:
    Bash

git clone <your-repository-url>
cd <repository-folder>

Create and activate a virtual environment (recommended):
Bash

python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

Install the dependencies:
Bash

    pip install -r requirements.txt

⚙️ Usage

1. Data Preprocessing

Before training, you must process the raw DICOM images into PNGs and split them into train/validation/test sets. Place your raw data (stage_2_train_labels.csv and stage_2_train_images directory) into a raw_data folder as specified in config.py.
Bash

python data_preprocessing.py

This will create a data directory with the processed images.

2. Model Training

To train the model, run the training script. The best model weights will be saved in the saved_models directory.
Bash

python train.py

3. Model Evaluation

To evaluate the trained model on the test set and see a sample CAM visualization, run the evaluation script.
Bash

python evaluate.py

4. Running the Web Application

To start the interactive web application, run the Flask app.
Bash

python app.py

Navigate to http://127.0.0.1:5000 in your web browser to upload an image and see the prediction.

🧠 Technical Details

    Data Preparation: The pipeline reads DICOM images, filters for "Pneumonia" and "Healthy" cases, and balances the classes to ensure an equal number of samples. Images are resized to 128x128 pixels. The data is split into 70% for training, 20% for validation, and 10% for testing.

    Model Architecture: The project uses a custom 10-layer CNN. It consists of four convolutional blocks with 3x3 filters and MaxPooling2D layers. A GlobalAveragePooling2D layer is used before the final Dense layer, which makes the model compatible with CAM-based visualizations.

    Training: The model is trained for up to 20 epochs using the Adam optimizer with a learning rate of 0.0001 and categorical_crossentropy as the loss function. Callbacks like EarlyStopping and ModelCheckpoint are used to save the best model and prevent overfitting.

💻 Technology Stack

    Backend: Flask

    Deep Learning: TensorFlow (Keras)

    Data Manipulation: Pandas, NumPy

    Image Processing: OpenCV, Pydicom

    Machine Learning: Scikit-learn
