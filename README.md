import cv2
import numpy as np
import matplotlib.pyplot as plt

# Завантаження зображення
image = cv2.imread('text_image.png', 0)  # Змінити на шлях до вашого зображення

# Бінаризація зображення
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Структурний елемент
kernel = np.ones((5, 5), np.uint8)

# Ерозія
eroded_image = cv2.erode(binary_image, kernel, iterations=1)

# Дилатація
dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

# Розмикання
opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

# Замикання
closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

# Границі
gradient_image = cv2.morphologyEx(binary_image, cv2.MORPH_GRADIENT, kernel)

# Відображення результатів
images = [binary_image, eroded_image, dilated_image, opened_image, closed_image, gradient_image]
titles = ['Original', 'Erosion', 'Dilation', 'Opening', 'Closing', 'Gradient']

plt.figure(figsize=(12, 8))
for i in range(len(images)):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()

# Збереження результатів
cv2.imwrite('eroded_image.png', eroded_image)
cv2.imwrite('dilated_image.png', dilated_image)
cv2.imwrite('opened_image.png', opened_image)
cv2.imwrite('closed_image.png', closed_image)
cv2.imwrite('gradient_image.png', gradient_image)
