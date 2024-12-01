import os
from tensorflow.keras.models import load_model

# Load the existing model
model = load_model(os.path.join(os.path.dirname(__file__), 'model', 'model.keras'))

# Save the model in HDF5 format
model.save(os.path.join(os.path.dirname(__file__), 'model', 'model.h5'))