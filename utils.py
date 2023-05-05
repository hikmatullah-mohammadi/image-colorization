import tensorflow as tf
from PIL import Image
from skimage.color import lab2rgb, rgb2lab
import numpy as np
import cv2

def is_grayscale(img):
    # Check if the image is in grayscale mode
    img = Image.fromarray(np.uint8(img))
    if img.mode == 'L':
        return True
    # Check if all channels are equal
    elif img.mode == 'RGB':
        r, g, b = img.split()
        if r == g == b:
            return True
    # If none of the above conditions are met, the image is not grayscale
    return False


# Preprocess the input image
def preprocess_input_image(img):
    # BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize the input image to 256x256
    img = cv2.resize(img, (256, 256))
    # Convert the input image to LAB color space
    img_lab = rgb2lab(img / 255.)
    # Extract the L channel
    L = img_lab[:,:,0]

    # # Normalize the input image to the range of -1 to 1
    L = L / 50 - 1
    return L.reshape(1, 256, 256, 1)


# Predict the colorized image
def colorize_image(autoencoder, X):
    Y_pred = autoencoder.predict(X)
    # re-transform normalization: change back from [-1, 1] to [-128, 128]
    Y_pred = Y_pred * 128 
    colorized_images = np.zeros((X.shape[0], 256, 256, 3))
    for i in range(X.shape[0]):
        # re-transform normalization: change back from [-1, 1] to [0, 100]
        l = X[i][:, :, 0] * 50 + 50 
        ab = Y_pred[i]
        colorized_images[i] = np.concatenate((l[:, :, np.newaxis], ab), axis=2)
        colorized_images[i] = lab2rgb(colorized_images[i])
    return colorized_images * 255.