import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
import warnings
warnings.filterwarnings('ignore')

from keras.models import Model
from keras.layers import Input, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#import segmentation_models as sm


# ============================================================
# Configuration
# ============================================================
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 1
BATCH_SIZE = 12
EPOCHS = 50


# ============================================================
# Paths
# ============================================================
train_image_dir = r"C:\Users\HP\python\ML_for_imaging_data\project_2\training\images"
train_mask_dir  = r"C:\Users\HP\python\ML_for_imaging_data\project_2\training\masks"

print(len(os.listdir(train_image_dir)), "images found.")
print("The program is running")


# ============================================================
# Function: Load TRAIN IMAGES + MASKS
# ============================================================
def load_images_and_masks(image_dir, mask_dir=None):
    image_ids = os.listdir(image_dir)

    X = np.zeros((len(image_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y = np.zeros((len(image_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32) if mask_dir else None

    for i, image_name in enumerate(image_ids):
        print(f"Loading image {i+1}/{len(image_ids)}: {image_name}")

        img_path = os.path.join(image_dir, image_name)
        img = imread(img_path, as_gray=True)
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X[i, ..., 0] = img.astype(np.float32) * 255

        if mask_dir:
            mask_path = os.path.join(mask_dir, image_name)
            mask = imread(mask_path, as_gray=True)
            mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            Y[i, ..., 0] = (mask > 0.5).astype(np.uint8)

    return X, Y


# ============================================================
# SMART AUTO-LOADING SYSTEM
# ============================================================
X_cache = "X_train.npy"
Y_cache = "Y_train.npy"

if os.path.exists(X_cache) and os.path.exists(Y_cache):
    print("Cached NumPy arrays found — loading them...")
    X_train = np.load(X_cache)
    Y_train = np.load(Y_cache)
    print("Loaded instantly!")
else:
    print("No cached arrays found — loading images from folders...")
    X_train, Y_train = load_images_and_masks(train_image_dir, train_mask_dir)

    print("Saving arrays for next runs...")
    np.save(X_cache, X_train)
    np.save(Y_cache, Y_train)
    print("Saved X_train.npy and Y_train.npy!")

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("Starting training...")


from sklearn.model_selection import train_test_split

X_train2, X_val, Y_train2, Y_val = train_test_split(
    X_train, Y_train, test_size=0.1, random_state=42
)


# ============================================================
# Loss functions
# ============================================================
def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    )


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    )


# ============================================================
# Build U-Net Model
# ============================================================
def build_unet():
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = Lambda(lambda x: x / 255)(inputs)

    def conv_block(input_tensor, num_filters, dropout_rate=0.1):
        x = Conv2D(num_filters, (3, 3), activation='elu',
                   padding='same', kernel_initializer='he_normal')(input_tensor)
        x = Dropout(dropout_rate)(x)
        x = Conv2D(num_filters, (3, 3), activation='elu',
                   padding='same', kernel_initializer='he_normal')(x)
        return x

    c1 = conv_block(s, 16)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 32)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 64, 0.2)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, 128, 0.2)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = conv_block(p4, 256, 0.3)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = conv_block(u6, 128, 0.2)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = conv_block(u7, 64, 0.2)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = conv_block(u8, 32, 0.1)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = conv_block(u9, 16, 0.1)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss=bce_dice_loss,
                  metrics=['accuracy', dice_coef])

    return model


model = build_unet()
model.summary()


# ============================================================
# Training
# ============================================================
filepath = r"C:\Users\HP\python\ML_for_imaging_data\project_2\best_unet_model.h5"

callbacks = [
    EarlyStopping(patience=5, verbose=1),
    ModelCheckpoint(filepath,
                    monitor='val_dice_coef',
                    save_best_only=True,
                    mode='max',
                    verbose=1)
]

history = model.fit(
    X_train2, Y_train2,
    validation_data=(X_val, Y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks
)


# ============================================================
# Plot results
# ============================================================
plt.plot(history.history['dice_coef'], label='train_dice')
plt.plot(history.history['val_dice_coef'], label='val_dice')
plt.title("Dice Score Over Epochs")
plt.legend()
plt.show()


# ============================================================
# Threshold tuning
# ============================================================
val_preds = model.predict(X_val)

def dice_np(y_true, y_pred, smooth=1e-6):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    intersection = np.sum(y_true * y_pred)
    return (2 * intersection + smooth) / (
        np.sum(y_true) + np.sum(y_pred) + smooth
    )

thresholds = np.linspace(0.05, 0.6, 19)
dice_scores = []

for t in thresholds:
    bin_preds = (val_preds >= t).astype(np.uint8)
    dice = dice_np(Y_val, bin_preds)
    dice_scores.append(dice)
    print(f"Threshold {t:.2f} → Dice {dice:.4f}")

best_t = thresholds[np.argmax(dice_scores)]
print(f"\n✅ Best threshold: {best_t:.2f}")

plt.plot(thresholds, dice_scores, marker='o')
plt.xlabel("Threshold")
plt.ylabel("Dice score")
plt.title("Threshold tuning")
plt.grid(True)
plt.show()
