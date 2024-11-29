import os
import logging
import numpy as np
import cupy as cp
import h5py
import cv2
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch.utils.data import DataLoader, Dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.mixed_precision import set_global_policy

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
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Exception in future: {e}")

def data_generator(hdf5_path, batch_size):
    with h5py.File(hdf5_path, 'r') as f:
        data = f['data']
        labels = f['labels']
        num_samples = data.shape[0]
        while True:
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                yield data[start:end], labels[start:end]

try:
    loaded = np.load(os.path.join(os.path.realpath(__file__), '..', 'data', 'quickdraw_dataset.npz'))
    data = loaded['data']
    labels = loaded['labels']
    create_hdf5_dataset(data, labels, os.path.join(os.path.realpath(__file__), '..', 'data', 'quickdraw_resized.hdf5'))
    print("Data and labels successfully loaded and resized!")
except Exception as e:
    print("Error loading data:", e)