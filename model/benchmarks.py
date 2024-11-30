import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.models import load_model
from sklearn.preprocessing import label_binarize

# Create the directory for saving benchmarks if it doesn't exist
benchmark_dir = os.path.join(os.path.dirname(__file__), 'mobilenet2.0_benchmarks')
os.makedirs(benchmark_dir, exist_ok=True)

# Load the data
loaded = np.load(os.path.join(os.path.dirname(__file__), 'data', 'quickdraw_dataset.npz'))
data = loaded['data']
labels = loaded['labels']

# Split the data into training and test sets with a 20% test size
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Preprocess the test data
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'model', 'model.keras')
model = load_model(model_path)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Predictions and confusion matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

y_true = y_test

conf_matrix = confusion_matrix(y_true, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)

# Plot and save the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
conf_matrix_plot_path = os.path.join(benchmark_dir, 'confusion_matrix.png')
plt.savefig(conf_matrix_plot_path)
plt.close()

# Classification report
class_report = classification_report(y_true, y_pred_classes, output_dict=True)
print("Classification Report:")
print(class_report)

# Plot and save the classification report
plt.figure(figsize=(10, 8))
sns.heatmap(pd.DataFrame(class_report).iloc[:-1, :].T, annot=True, cmap='Blues')
plt.title('Classification Report')
class_report_plot_path = os.path.join(benchmark_dir, 'classification_report.png')
plt.savefig(class_report_plot_path)
plt.close()
