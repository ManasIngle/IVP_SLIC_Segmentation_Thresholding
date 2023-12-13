import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter

def otsu_thresholding(image):
    # Compute histogram
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])

    # Normalize histogram
    hist_norm = hist / hist.sum()

    # Variables to store optimum threshold and maximum variance
    max_variance, optimum_threshold = 0, 0

    # Compute cumulative distribution and mean values
    cum_dist = np.cumsum(hist_norm)
    cum_mean = np.cumsum(np.multiply(hist_norm, np.arange(256)))

    for t in range(1, 256):
        # Class probabilities
        w1 = cum_dist[t]
        w2 = 1 - w1

        # Class means
        mean1 = cum_mean[t] / w1
        mean2 = (cum_mean[255] - cum_mean[t]) / w2

        # Class variances
        var1 = np.sum((np.arange(t) - mean1) ** 2 * hist_norm[:t]) / w1
        var2 = np.sum((np.arange(t + 1, 256) - mean2) ** 2 * hist_norm[t + 1:]) / w2

        # Compute within-class variance
        within_class_variance = w1 * var1 + w2 * var2

        # Update optimum threshold if within-class variance is greater
        if within_class_variance > max_variance:
            max_variance = within_class_variance
            optimum_threshold = t

    return optimum_threshold

def segment_image(image_path):
    # Load the image
    img = Image.open(image_path).convert('L')  

    # Convert the image to a NumPy array
    img_array = np.array(img)

    img_smoothed = gaussian_filter(img_array, sigma=1.5)

    # Apply Otsu's thresholding
    threshold = otsu_thresholding(img_smoothed)

    # Create a binary mask using the threshold
    binary_mask = (img_smoothed > threshold).astype(np.uint8) * 255

    # Refine the binary mask (optional)
    binary_mask = binary_mask * (img_smoothed > img_smoothed.mean())

    # Display the original and segmented images
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(img_array, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 3, 3)
    plt.imshow(binary_mask, cmap='gray')
    plt.title('Segmented Image')

    plt.show()

image_path = 'image.png'
segment_image(image_path)
