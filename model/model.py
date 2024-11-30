import os
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import pandas as pd

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
    loaded = np.load(os.path.join(os.path.realpath(__file__), '..', 'data', 'quickdraw_dataset.npz'))
    data = loaded['data']
    labels = loaded['labels']
    print("Data and labels successfully loaded and resized!")
except Exception as e:
    print("Error loading data:", e)
    quit(1)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

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

input_shape = (28, 28, 3)

X_train_rgb = np.repeat(X_train, 3, axis=-1)
X_test_rgb = np.repeat(X_test, 3, axis=-1)

print("Training data shape:", X_train_rgb.shape)
print("Testing data shape:", X_test_rgb.shape)

train_gen = datagen.flow(X_train_rgb, y_train, batch_size=32)

def create_model(input_shape, num_classes):
    input_layer = Input(shape=input_shape)

    base_model = MobileNetV2(input_tensor=input_layer, include_top=False, weights='imagenet', alpha=0.5)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

num_classes = 345
model = create_model(input_shape, num_classes)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6
)

checkpoint = ModelCheckpoint(
    filepath=os.path.join(os.path.dirname(__file__), 'model', 'accelerated_model.keras'),
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)

try:
    history = model.fit(
        train_gen,
        steps_per_epoch=len(X_train_rgb) // 32,
        epochs=10,
        validation_data=(X_test_rgb, y_test),
        callbacks=[early_stopping, reduce_lr, checkpoint]
    )
except KeyboardInterrupt:
    print("Training interrupted... saving model now...")


tf.keras.models.save_model(
    model, os.path.join(os.path.dirname(__file__), 'model', 'accelerated_model.keras'), overwrite=True,
    include_optimizer=True, save_format=None,
    signatures=None, options=None)

with custom_object_scope({'Cast': tf.keras.layers.Layer}):
    model = load_model(os.path.join(os.path.dirname(__file__), 'model', 'accelerated_model.keras'))

output_dir = os.path.join(os.path.dirname(__file__), 'mobilenet2.0_benchmarks')
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'benchmarks_loss.png'))
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'benchmarks_accuracy.png'))
plt.close()

test_loss, test_accuracy = model.evaluate(X_test_rgb, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

y_pred = model.predict(X_test_rgb)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

conf_matrix = confusion_matrix(y_true, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close()

class_report = classification_report(y_true, y_pred_classes, output_dict=True)
print("Classification Report:")
print(class_report)

plt.figure(figsize=(10, 7))
sns.heatmap(pd.DataFrame(class_report).iloc[:-1, :].T, annot=True, cmap='Blues')
plt.title('Classification Report')
plt.xlabel('Metrics')
plt.ylabel('Classes')
plt.savefig(os.path.join(output_dir, 'classification_report.png'))
plt.close()

test_loss, test_accuracy = model.evaluate(X_test_rgb, y_test, verbose=2)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
