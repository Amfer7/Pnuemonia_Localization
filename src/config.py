# Configuration settings for the project
IMG_SIZE = 128
RAW_DATA_PATH = 'raw_data'
PROCESSED_DATA_PATH = 'data'
TRAIN_PATH = f'{PROCESSED_DATA_PATH}/train'
VALIDATION_PATH = f'{PROCESSED_DATA_PATH}/validation'
TEST_PATH = f'{PROCESSED_DATA_PATH}/test'
TRAIN_CSV = f'{RAW_DATA_PATH}/stage_2_train_labels.csv'
TRAIN_IMAGES_DIR = f'{RAW_DATA_PATH}/stage_2_train_images'

# Data split ratios from the paper [cite: 41]
VALIDATION_SPLIT = 0.20
TEST_SPLIT = 0.10