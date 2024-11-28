import tensorflow as tf
import numpy as np
import os
import cv2
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.mixed_precision import set_global_policy
import cupy as cp
import h5py
import logging
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Num GPUs Available: {len(gpus)}")
    for gpu in gpus:
        print(f"GPU: {gpu}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is configured for use.")
    except RuntimeError as e:
        print("Error in setting up GPU:", e)
else:
    print("No GPUs found. Please check your CUDA and cuDNN installations.")

try:
    set_global_policy('mixed_float16')
except Exception as e:
    print("Error setting mixed precision:", e)

def resize_image(image):
    image_gpu = cp.asarray(image, dtype=cp.float32)
    resized_image_gpu = cp.resize(image_gpu, (28, 28))
    return cp.asnumpy(resized_image_gpu)

"""
# Create HDF5 dataset with threading and CuPy
def create_hdf5_dataset(data, labels, hdf5_path):
    with h5py.File(hdf5_path, 'w') as f:
        f.create_dataset('data', (len(data), 28, 28, 1), dtype='float32')
        f.create_dataset('labels', data=labels, dtype='int')
        
        def process_and_store(start, end, thread_id):
            num_images = end - start
            logging.info(f"Starting thread {thread_id}, with {num_images} images")
            try:
                resized_images = []
                for img in data[start:end]:
                    resized_img = resize_image(img)
                    resized_img = np.expand_dims(resized_img, axis=-1)
                    resized_images.append(resized_img)
                f['data'][start:end] = np.array(resized_images, dtype='float32')
            except Exception as e:
                logging.error(f"Error in thread {thread_id}: {e}")
            finally:
                logging.info(f"Ending thread {thread_id}, with {num_images} images")

        batch_size = 1000
        futures = []
        with ThreadPoolExecutor() as executor:
            for thread_id, start in enumerate(range(0, len(data), batch_size)):
                end = min(start + batch_size, len(data))
                futures.append(executor.submit(process_and_store, start, end, thread_id))
            
            # Ensure all futures complete
            for future in as_completed(futures):
                try:
                    future.result()  # This will raise any exceptions caught during execution
                except Exception as e:
                    logging.error(f"Exception in future: {e}")
"""

try:
    loaded = np.load(os.path.join(os.path.realpath(__file__), '..', 'data', 'quickdraw_dataset.npz'))
    data = loaded['data']
    labels = loaded['labels']
    # create_hdf5_dataset(data, labels, os.path.join(os.path.realpath(__file__), '..', 'data', 'quickdraw_resized.hdf5'))
    print("Data and labels successfully loaded and resized!")
except Exception as e:
    print("Error loading data:", e)
"""
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
"""

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

num_samples = X_train.shape[0]

total_elements = num_samples * 28 * 28
print(total_elements)

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
train_gen = datagen.flow(X_train, y_train, batch_size=32)

"""
# Define a more complex model architecture
def create_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        
        # First block
        Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Second block
        Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Third block
        Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Fully connected layers
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax', dtype='float32')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
"""

def create_model(input_shape, num_classes):
    input_layer = Input(shape=input_shape)

    base_model = MobileNetV2(input_tensor=input_layer, include_top=False, weights=None, alpha=0.5)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

input_shape = (28, 28, 1)
num_classes = 345
model = create_model(input_shape, num_classes)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

model.fit(
    train_gen,
    steps_per_epoch=len(X_train) // 32,
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr]
)

# Save the model using the recommended Keras format
model.save(os.path.join(os.path.realpath(__file__), '..', 'model', 'model.keras'))

# Load the model with custom object scope if necessary
with custom_object_scope({'Cast': tf.keras.layers.Layer}):  # Replace 'Cast' with the actual custom layer if needed
    model = load_model(os.path.join(os.path.realpath(__file__), '..', 'model', 'model.keras'))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
