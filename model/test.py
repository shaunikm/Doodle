import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Define paths
dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'quickdraw_dataset.npz')
model_path = os.path.join(os.path.dirname(__file__), 'model', 'model.keras')
predictions_path = os.path.join(os.path.dirname(__file__), 'data', 'categories.txt')

# Load the dataset
loaded = np.load(dataset_path)
data = loaded['data']
labels = loaded['labels']

# Load the model
model = load_model(model_path)

# Load categories
with open(predictions_path, 'r') as file:
    categories = [line.strip() for line in file.readlines()]

# Prepare the data
index = 100000  # Change this index to view different images
image = data[index].reshape(28, 28, 1).astype('float32') / 255.0
image_rgb = np.repeat(image, 3, axis=-1).reshape(1, 28, 28, 3)

# Print the input array
print("Input Array:")
print(data[index])
print(labels)
print(index)

# Print unique labels
unique_labels = np.unique(labels)
print("Unique Labels:")
print(unique_labels)

# Get the model's prediction
prediction = model.predict(image_rgb)
predicted_index = np.argmax(prediction, axis=1)[0]
predicted_category = categories[predicted_index]

# Display the image, ground truth, and prediction
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"Ground Truth: {categories[labels[index]]}, Prediction: {predicted_category}")
plt.axis('off')
plt.show()