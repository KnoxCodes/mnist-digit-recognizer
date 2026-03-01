import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np


st.set_page_config(page_title="MNIST AI", page_icon="🧠", layout="centered")


class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

model = ANN()
model.load_state_dict(torch.load("mnist_model.pth", map_location="cpu"))
model.eval()


st.markdown("<h1 style='text-align:center;'>🧠 MNIST Digit Recognizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Draw a digit and let the AI predict it</p>", unsafe_allow_html=True)

st.write("")


canvas_result = st_canvas(
    fill_color="black",
    stroke_width=14,
    stroke_color="white",
    background_color="#000000",
    height=360,
    width=435,
    drawing_mode="freedraw",
    key="canvas",
)

col1, col2 = st.columns(2)

predict_clicked = col1.button("🔍 Predict")
clear_clicked = col2.button("🗑 Clear")

if clear_clicked:
    st.rerun()


if predict_clicked:

    if canvas_result.image_data is not None:

        img = Image.fromarray(
            (canvas_result.image_data[:, :, 0]).astype("uint8")
        )

        img = img.resize((28, 28))

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, 1).item()

        st.write("")
        st.success(f"🎯 Prediction: {prediction}")

        st.write("### Confidence Levels")
        st.progress(float(torch.max(probabilities)))

        probs = probabilities.numpy()[0]

        chart_data = {
            "Digit": list(range(10)),
            "Confidence": probs
        }

        st.bar_chart(chart_data, x="Digit", y="Confidence")