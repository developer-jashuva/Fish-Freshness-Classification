# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # Define dataset paths
# train_dir = "./dataset/train"
# test_dir = "./dataset/test"

# # Define image size and batch size
# IMG_SIZE = (128, 128)
# BATCH_SIZE = 32

# # Load the dataset (Same as before)
# train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
# test_datagen = ImageDataGenerator(rescale=1./255)

# train_data = train_datagen.flow_from_directory(
#     train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", subset="training"
# )
# val_data = train_datagen.flow_from_directory(
#     train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", subset="validation"
# )
# test_data = test_datagen.flow_from_directory(
#     test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
# )

# # Define CNN Model
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
#     MaxPooling2D(2, 2),

#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),

#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),

#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(len(train_data.class_indices), activation='softmax')  # Output layer for categories
# ])

# # Compile Model
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Train Model
# history = model.fit(train_data, validation_data=val_data, epochs=10)

# # Save Model
# model.save("../fish_freshness_cnn.h5")

# print("✅ Model Training Completed & Saved as 'fish_freshness_cnn.h5'!")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

# Define dataset paths
train_dir = "./dataset/train"
test_dir = "./dataset/test"

# Define image size and batch size
IMG_SIZE = (128, 128)
BATCH_SIZE = 16

# Data Augmentation (For Oversampling)
train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=30, width_shift_range=0.2, height_shift_range=0.2,
    horizontal_flip=True, zoom_range=0.2, brightness_range=[0.8, 1.2], validation_split=0.2
)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load dataset
train_data = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", subset="training"
)
val_data = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", subset="validation"
)
test_data = test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)

# Compute Class Weights
class_labels = list(train_data.class_indices.keys())
class_counts = np.array([len(os.listdir(os.path.join(train_dir, label))) for label in class_labels])
total_samples = np.sum(class_counts)
class_weights = {i: total_samples / (len(class_counts) * class_counts[i]) for i in range(len(class_counts))}

print("Class Weights:", class_weights)

# Define Improved CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_labels), activation='softmax')  # Output layer
])

# Compile Model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train Model with Class Weights
history = model.fit(train_data, validation_data=val_data, epochs=10, class_weight=class_weights)

# Save Model
model.save("./fish_freshness_cnn.h5")

print("✅ Model Training Completed & Saved as 'fish_freshness_cnn.h5'!")

