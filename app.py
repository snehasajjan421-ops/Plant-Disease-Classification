import json
import os
from PIL import Image
import requests
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms

st.set_page_config(
    page_title="Plant Disease Classifier", page_icon="🌱", layout="centered"
)

st.title("🌱 Plant Disease Classification System")
st.write("Upload a leaf image to detect potential plant diseases in real-time.")

device = torch.device("cpu")

WEIGHTS_FILE = "plant_disease_model.1.pth"
WEIGHTS_URL = "https://github.com/snehasajjan421-ops/Plant-Disease-Classification/releases/download/v1.0/plant_disease_model.1.pth"


# Robust downloader using requests to follow GitHub Release redirects
def download_weights(url, output_path):
  headers = {"User-Agent": "Mozilla/5.0"}
  response = requests.get(url, headers=headers, stream=True, allow_redirects=True)
  response.raise_for_status()
  with open(output_path, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
      if chunk:
        f.write(chunk)


if not os.path.exists(WEIGHTS_FILE) or os.path.getsize(WEIGHTS_FILE) < 1000:
  with st.spinner("Downloading trained model weights... please wait"):
    download_weights(WEIGHTS_URL, WEIGHTS_FILE)


@st.cache_resource
def load_model():
  with open("class_names.json", "r") as f:
    class_names = json.load(f)

  model = models.resnet18(weights=None)
  model.fc = nn.Linear(model.fc.in_features, len(class_names))
  model.load_state_dict(
      torch.load(WEIGHTS_FILE, map_location=device, weights_only=True)
  )
  model.to(device)
  model.eval()
  return model, class_names


try:
  model, class_names = load_model()
  model_ready = True
except Exception as e:
  st.error(f"Error loading model: {e}")
  model_ready = False

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ),
])

uploaded_file = st.file_uploader(
    "Choose a leaf image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and model_ready:
  image = Image.open(uploaded_file).convert("RGB")
  st.image(image, caption="Uploaded Leaf", use_container_width=True)

  tensor_img = transform(image).unsqueeze(0).to(device)
  with torch.no_grad():
    output = model(tensor_img)
    probs = torch.nn.functional.softmax(output[0], dim=0)

  top_prob, top_idx = torch.topk(probs, 3)

  st.subheader("Predictions:")
  for i in range(3):
    idx = top_idx[i].item()
    confidence = top_prob[i].item() * 100
    st.write(f"**{class_names[idx]}**: {confidence:.2f}%")
    st.progress(float(top_prob[i].item()))
