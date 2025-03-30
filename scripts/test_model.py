# import tensorflow as tf
# import numpy as np
# import cv2
# import os

# # Load the trained model
# model = tf.keras.models.load_model("./fish_freshness_cnn.h5")

# # Define class labels (based on the dataset structure)
# class_labels = ["Fresh_Eyes", "Fresh_Gills", "Nonfresh_Eyes", "Nonfresh_Gills"]

# # Function to preprocess image
# def preprocess_image(img_path):
#     img = cv2.imread(img_path)
#     img = cv2.resize(img, (128, 128))  # Resize to match training input size
#     img = img / 255.0  # Normalize
#     img = np.expand_dims(img, axis=0)  # Expand dimensions for model
#     return img

# # Test on a sample image
# test_image_path = "./dataset/test/Fresh_Gills/sample.jpg"  # Change with your actual test image path
# img = preprocess_image(test_image_path)

# # Make prediction
# prediction = model.predict(img)
# predicted_class = np.argmax(prediction)

# # Show result
# print(f"Predicted Class: {class_labels[predicted_class]}")
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import os
from PIL import Image

# Load the pre-trained model
model = models.efficientnet_b0(pretrained=False)  # Using EfficientNet-B0
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 4)  # 4 output classes

# Load the trained weights
model.load_state_dict(torch.load("./fish_freshness_cnn.h5", map_location=torch.device('cpu')))
model.eval()  # Set model to evaluation mode

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match the model input size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image_path, model):
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    class_names = ["Fresh_Eyes", "Fresh_Gills", "Nonfresh_Eyes", "Nonfresh_Gills"]
    return class_names[predicted.item()]

# Set the directory for test images
test_dir = "./dataset/test/Fresh_Eyes"  # Replace with the actual path
image_files = [f for f in os.listdir(test_dir) if f.endswith(('jpg', 'png', 'jpeg'))]

correct = 0
total = len(image_files)

# Run predictions
for img_file in image_files:
    img_path = os.path.join(test_dir, img_file)
    predicted_class = predict(img_path, model)
    
    print(f"Image: {img_file} → Predicted: {predicted_class}")

# Display accuracy if ground-truth labels are available
print(f"Processed {total} images.")
