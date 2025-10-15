import os
import pandas as pd
import pydicom
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from src.config import *

def create_dirs():
    """Creates directories for the processed data."""
    for path in [PROCESSED_DATA_PATH, TRAIN_PATH, VALIDATION_PATH, TEST_PATH]:
        os.makedirs(os.path.join(path, 'Pneumonia'), exist_ok=True)
        os.makedirs(os.path.join(path, 'Healthy'), exist_ok=True)

def process_and_save_image(patient_id, output_dir, label):
    """Reads a DICOM image, resizes it, and saves it as a PNG."""
    dicom_path = os.path.join(TRAIN_IMAGES_DIR, f'{patient_id}.dcm')
    output_path = os.path.join(output_dir, label, f'{patient_id}.png')
    try:
        dcm = pydicom.dcmread(dicom_path)
        # Resize image to 128x128 pixels [cite: 55]
        img_resized = cv2.resize(dcm.pixel_array, (IMG_SIZE, IMG_SIZE))
        # Normalize to 0-255 and save as PNG
        img_normalized = cv2.normalize(img_resized, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(output_path, img_normalized.astype('uint8'))
    except FileNotFoundError:
        # Some images in the CSV may not exist in the folder, so we skip them.
        print(f"ERROR: Could not find input file: {dicom_path}")

def main():
    print("Starting data preprocessing...")
    create_dirs()
    labels_df = pd.read_csv(TRAIN_CSV)
    
    # Filter for only Pneumonia (Target=1) and Healthy (Target=0) cases, removing others [cite: 39]
    pneumonia_df = labels_df[labels_df['Target'] == 1].drop_duplicates('patientId')
    healthy_df = labels_df[labels_df['Target'] == 0].drop_duplicates('patientId')
    
    # Balance the dataset to have an equal number of samples [cite: 40]
    healthy_df = healthy_df.sample(n=len(pneumonia_df), random_state=42)
    
    pneumonia_df['label'] = 'Pneumonia'
    healthy_df['label'] = 'Healthy'
    combined_df = pd.concat([pneumonia_df, healthy_df])
    
    # Split data into 70/20/10 for train/validation/test [cite: 41]
    train_val_df, test_df = train_test_split(combined_df, test_size=TEST_SPLIT, random_state=42, stratify=combined_df['label'])
    relative_val_split = VALIDATION_SPLIT / (1 - TEST_SPLIT)
    train_df, val_df = train_test_split(train_val_df, test_size=relative_val_split, random_state=42, stratify=train_val_df['label'])

    datasets = {'train': (train_df, TRAIN_PATH), 'validation': (val_df, VALIDATION_PATH), 'test': (test_df, TEST_PATH)}

    for name, (df, path) in datasets.items():
        print(f"Processing {name} images ({len(df)} total)...")
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            process_and_save_image(row['patientId'], path, row['label'])
    print("Preprocessing complete.")

if __name__ == "__main__":
    main()