import tensorflow as tf
import numpy as np
import os
import cv2
import torch
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader, Dataset
import psutil
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.mixed_precision import set_global_policy
import cupy as cp
import h5py

# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Num GPUs Available: {len(gpus)}")
    for gpu in gpus:
        print(f"GPU: {gpu}")
else:
    print("No GPUs found. Please check your CUDA and cuDNN installations.")

# Set memory growth for the GPU to prevent memory issues
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is configured for use.")
    except RuntimeError as e:
        print("Error in setting up GPU:", e)

# Enable mixed precision
try:
    set_global_policy('mixed_float16')
except Exception as e:
    print("Error setting mixed precision:", e)

# Function to resize images using CuPy
def resize_image(image):
    image_gpu = cp.asarray(image, dtype=cp.float32)
    resized_image_gpu = cp.resize(image_gpu, (256, 256))
    return cp.asnumpy(resized_image_gpu)

# Create HDF5 dataset with threading and CuPy
def create_hdf5_dataset(data, labels, hdf5_path):
    with h5py.File(hdf5_path, 'w') as f:
        f.create_dataset('data', (len(data), 256, 256, 1), dtype='float32')
        f.create_dataset('labels', data=labels, dtype='int')
        
        def process_and_store(start, end):
            resized_images = []
            for img in data[start:end]:
                resized_img = resize_image(img.reshape(28, 28))
                resized_images.append(resized_img[..., np.newaxis] / 255.0)
            f['data'][start:end] = np.array(resized_images, dtype='float32')

        batch_size = 1000
        with ThreadPoolExecutor() as executor:
            for start in range(0, len(data), batch_size):
                end = min(start + batch_size, len(data))
                executor.submit(process_and_store, start, end)

# Load data and create HDF5 file
try:
    loaded = np.load(os.path.join(os.path.realpath(__file__), '..', 'data', 'quickdraw_dataset.npz'))
    data = loaded['data']
    labels = loaded['labels']
    create_hdf5_dataset(data, labels, os.path.join(os.path.realpath(__file__), '..', 'data', 'quickdraw_resized.hdf5'))
    print("Data and labels successfully loaded and resized!")
except Exception as e:
    print("Error loading data:", e)

# Data generator
def data_generator(hdf5_path, batch_size):
    with h5py.File(hdf5_path, 'r') as f:
        data = f['data']
        labels = f['labels']
        num_samples = data.shape[0]
        while True:
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                yield data[start:end], labels[start:end]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

num_samples = X_train.shape[0]
num_samples

total_elements = num_samples * 256 * 256
print(total_elements)

# Data augmentation for training data
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
train_gen = datagen.flow(X_train, y_train, batch_size=32)

# Define the model
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax', dtype='float32')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Create and train the model
num_classes = 345
model = create_model((256, 256, 1), num_classes)

# Use the data generator for training
train_data_gen = data_generator('data/quickdraw_resized.hdf5', batch_size=32)
model.fit(train_data_gen, steps_per_epoch=len(X_train) // 32, epochs=10, validation_data=(X_test, y_test))
