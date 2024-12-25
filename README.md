import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the uploaded image
image_path = "/mnt/data/Знімок екрана 2024-12-25 231712.png"
image = cv2.imread(image_path, 0)

# Binary thresholding
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Structuring element
kernel = np.ones((5, 5), np.uint8)

# Morphological operations
eroded_image = cv2.erode(binary_image, kernel, iterations=1)
dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
gradient_image = cv2.morphologyEx(binary_image, cv2.MORPH_GRADIENT, kernel)

# Plot and save results
results = {
    "Original": binary_image,
    "Erosion": eroded_image,
    "Dilation": dilated_image,
    "Opening": opened_image,
    "Closing": closed_image,
    "Gradient": gradient_image
}

# Save results and show
output_paths = {}
for title, img in results.items():
    output_path = f"/mnt/data/{title.lower()}_image.png"
    cv2.imwrite(output_path, img)
    output_paths[title] = output_path

output_paths

