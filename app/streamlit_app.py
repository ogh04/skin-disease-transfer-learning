import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Melanoma", "Carcinoma", "Eczema"]
IMG_SIZE = 224
MODEL_PATH = "models/efficientnet_b4_best.pth"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = models.efficientnet_b4(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ---------------- PREPROCESSING ----------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def predict(image: Image.Image):
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
    pred_idx = torch.argmax(probs).item()
    return CLASS_NAMES[pred_idx], probs.cpu().numpy()

# ---------------- UI ----------------
st.set_page_config(page_title="Skin Disease Detection", page_icon="🩺")
st.title("🩺 Détection de Maladies Cutanées")
st.write("Upload une image de lésion cutanée pour obtenir une prédiction (Melanoma / Carcinoma / Eczema).")

uploaded_file = st.file_uploader("Choisis une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image uploadée", use_column_width=True)

    with st.spinner("Analyse en cours..."):
        predicted_class, probs = predict(image)

    st.success(f"**Diagnostic prédit : {predicted_class}**")
    st.write(f"Confiance : **{probs.max() * 100:.2f}%**")

    st.subheader("Probabilités par classe")
    for class_name, prob in zip(CLASS_NAMES, probs):
        st.write(f"{class_name}: {prob * 100:.2f}%")
        st.progress(float(prob))

    st.warning("⚠️ Cet outil est à but éducatif/démonstratif uniquement, il ne remplace pas un diagnostic médical professionnel.")