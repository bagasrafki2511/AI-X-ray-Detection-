import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import load_model
from gradcam import GradCAM
import cv2

st.set_page_config(page_title="X-ray AI Detection", layout="centered")

st.title("AI Deteksi Penyakit dari X-ray")
st.caption("⚠️ AI ini hanya sebagai decision support, bukan diagnosis final.")

model = load_model()
target_layer = model.model.features[-1]
cam_generator = GradCAM(model, target_layer)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

uploaded_file = st.file_uploader("Upload gambar X-ray", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar X-ray", width=300)

    input_tensor = transform(image).unsqueeze(0)

# Menentukan Tubercolosis atau Normalnya
    with torch.no_grad():
        outputs = model(input_tensor) #Hasil/Skor
        probs = torch.softmax(outputs, dim=1) 
        confidence, pred = torch.max(probs, 1)

    labels = ["Normal", "Tubercolosis"]
    st.subheader("Hasil Prediksi")
    st.write(f"**Prediksi:** {labels[pred.item()]}")
    st.write(f"**Confidence:** {confidence.item()*100:.2f}%")

    cam = cam_generator.generate(input_tensor, pred.item())
    cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    img = np.array(image.resize((224, 224)))
    heatmap = cv2.addWeighted(img, 0.6, cam, 0.4, 0)

    st.subheader("Heatmap (Grad-CAM)")
    st.image(heatmap, caption="Area penting menurut AI")