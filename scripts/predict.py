import torch
import torchvision.models as models  # ✅ Add this line
import torch.nn as nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# Load the saved model
model = models.efficientnet_b0(pretrained=False)

# Get the number of input features for the classifier
num_ftrs = model.classifier[1].in_features  # ✅ Define num_ftrs properly


model.classifier[1] = nn.Linear(num_ftrs, 4)
model.load_state_dict(torch.load("efficientnet_fish.pth"))

# Modify the classifier to match the number of output classes (4 classes: Fresh_Eyes, Fresh_Gills, etc.)
model.classifier[1] = nn.Linear(num_ftrs, 4)  # ✅ Now num_ftrs is defined

# Load the test dataset (Assuming it's in DataLoader format)
test_loader = torch.load("./efficientnet_fish.pth")  # Load saved test data if available

y_true = []
y_pred = []

model.eval()  # Set model to evaluation mode
with torch.no_grad():
    for batch in test_loader:
        inputs = batch[0]  # Features (images)
        labels = batch[1]  # True class labels

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.numpy())


# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
