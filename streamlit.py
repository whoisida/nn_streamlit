import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import requests
from io import BytesIO
import torchvision.models as models
import torch.nn as nn

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st

# Create an instance of the ResNet18 model
model = models.resnet18(pretrained=True)
# Modify the fully connected layer to match your output classes (11 classes in this case)
model.fc = nn.Linear(in_features=512, out_features=11, bias=True)

# Load the pre-trained weights
model.load_state_dict(torch.load('resnet18_model.pt', map_location=torch.device('cpu')))
model.eval()

# Define the transformation to be applied to the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

labels = {1: 'fogsmog', 4: 'hail', 0: 'dew', 9: 'sandstorm', 8: 'rime', 2: 'frost', 7: 'rainbow', 3: 'glaze', 6: 'rain', 10: 'snow', 5: 'lightning'}

st.write("""
# App for the weather classification based on photo
""")

file = st.file_uploader("Put your photo here, use jpg", type=["jpg"])

if file is not None:
    # Open the image and apply the transformation
    image = Image.open(file).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        # Forward pass through the model
        outputs = model(image)

        # Get the predicted class
        _, predicted = torch.max(outputs, 1)
        predicted_class = labels[predicted.item()]

        st.image(image.squeeze(0).numpy().transpose(1, 2, 0), caption='Your photo', use_column_width=True)
        st.write(f'Predicted class by model: {predicted_class}')
