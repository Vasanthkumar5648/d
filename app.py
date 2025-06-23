import streamlit as st
import cv2
import torch
import os
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tempfile import NamedTemporaryFile

# Load YOLOv5 model
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

model = load_model()

# Create output folders
os.makedirs("output/frames", exist_ok=True)
os.makedirs("output/annotated_frames", exist_ok=True)

st.title("🎯 Detection Summary Engine")
st.markdown("Upload a short `.mp4` video for object detection using YOLOv5.")
