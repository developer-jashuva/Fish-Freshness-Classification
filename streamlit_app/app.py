import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models

# Load the trained model
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 4)
    model.load_state_dict(torch.load("./models/model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Define class labels
class_labels = ['Fresh_Eyes', 'Fresh_Gills', 'Nonfresh_Eyes', 'Nonfresh_Gills']

# Define preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize image
        transforms.ToTensor(),          # Convert to tensor
        transforms.Normalize([0.5], [0.5])  # Normalize
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Streamlit UI
st.title("Fish Freshness Classification 🐟")
st.write("Upload an image of fish parts to check freshness.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_labels[predicted.item()]

    st.write(f"### Predicted Class: **{predicted_class}**")
