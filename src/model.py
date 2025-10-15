from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, BatchNormalization, Activation
from src.config import IMG_SIZE

def build_pneumonia_cnn():
    """
    Builds the 10-layer CNN model as described in the paper[cite: 5, 71].
    The structure is made compatible with CAM by using a GAP layer before the final Dense layer.
    """
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Using 3x3 filters is a key detail for better IoU scores [cite: 173]
    # Block 1 (2 Conv layers)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Block 2 (2 Conv layers)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Block 3 (3 Conv layers)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Block 4 (3 Conv layers)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    # The final convolutional layer before GAP, which is used for CAM
    x = Conv2D(512, (3, 3), padding='same', activation='relu', name='last_conv_layer')(x)

    # CAM-compatible output layers 
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(2, activation='softmax', name='predictions')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model