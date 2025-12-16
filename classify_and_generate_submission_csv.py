from skimage.io import imsave
import numpy as np
import pandas as pd
import os
from skimage.io import imread
from skimage.transform import resize
import warnings
warnings.filterwarnings('ignore')
from keras.models import load_model

# Configuration
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 1
BATCH_SIZE = 12
EPOCHS = 50
BEST_THRESHOLD = 0.48

test_image_dir = r"C:\Users\HP\python\ML_for_imaging_data\project_2\Testing\Images"

# Load data
def load_images(image_dir):
    image_ids = os.listdir(image_dir)
    X = np.zeros((len(image_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)

    for i, image_name in enumerate(image_ids):
        img_path = os.path.join(image_dir, image_name)
        img = imread(img_path, as_gray=True)
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X[i, ..., 0] = img.astype(np.float32) * 255

    return X

X_test = load_images(test_image_dir)
print("min/max of first image:", X_test[0].min(), X_test[0].max())

modelpath = r"C:\Users\HP\python\ML_for_imaging_data\project_2\best_unet_model.h5"

# Predict
model = load_model(modelpath, compile=False, safe_mode=False)
test_preds = model.predict(X_test)
thresholded_preds = (test_preds >= BEST_THRESHOLD).astype(np.uint8)

# Output directory for predicted masks
output_dir = r"C:\Users\HP\python\ML_for_imaging_data\project_2\predicted_masks" # For predicted masks but also for the csv submission file
os.makedirs(output_dir, exist_ok=True)

def rle_encoding(x):
    dots = np.where(x.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
        
    return ' '.join(map(str, run_lengths))

submission = []

# Save each predicted mask
image_ids = os.listdir(test_image_dir)
for i in range(len(thresholded_preds)):
    mask = thresholded_preds[i, ..., 0] * 255  # Remove extra dimensions, scale to 0–255
    mask_path = os.path.join(output_dir, f"mask_{i:03d}.png")
    imsave(mask_path, mask.astype(np.uint8))
    
    rle = rle_encoding((mask/255)> BEST_THRESHOLD)

    submission.append((i+1, rle))

#
#
#

df_sub = pd.DataFrame(submission, columns=['id', 'rle'])
df_sub.to_csv(os.path.join(output_dir, 'submission.csv'), index=False)
df_sub.head()
print("Submission saved successfully.")

#
#
#

import matplotlib.pyplot as plt
plt.imshow(test_preds[0, ..., 0], cmap='gray')
plt.colorbar()
plt.show()

#
#
#

import matplotlib.pyplot as plt

# Pick an image index to visualize
i = 0

# Rescale original image to [0, 1] for display (if it's in 0–255)
original = X_test[i, ..., 0] / 255.0 if X_test.max() > 1 else X_test[i, ..., 0]

# Predicted mask (before thresholding)
pred_mask = test_preds[i, ..., 0]

# Thresholded mask
binary_mask = (pred_mask >= BEST_THRESHOLD).astype(np.uint8)

# Plot overlay
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(original, cmap='gray')
plt.title("Original Test Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(pred_mask, cmap='viridis')
plt.title("Raw Prediction")
plt.colorbar()
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(original, cmap='gray')
plt.imshow(binary_mask, cmap='Reds', alpha=0.5)
plt.title("Overlayed Prediction")
plt.axis('off')

plt.tight_layout()
plt.show()
