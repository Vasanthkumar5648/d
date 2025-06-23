import streamlit as st
import cv2
import torch
import os
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tempfile import NamedTemporaryFile
from io import BytesIO
import numpy as np

# Load YOLOv5 model
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.eval()
    return model

model = load_model()

# Create output folders
os.makedirs("output/frames", exist_ok=True)
os.makedirs("output/annotated_frames", exist_ok=True)

st.title("🎯 Detection Summary Engine")
st.markdown("Upload a short `.mp4` video for object detection using YOLOv5.")

uploaded_file = st.file_uploader("Upload MP4 Video", type=["mp4"])

if uploaded_file:
    with NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(uploaded_file.read())
        temp_path = temp_video.name

    cap = cv2.VideoCapture(temp_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    st.success(f"Video Loaded: {int(duration)} seconds, {int(fps)} FPS, {total_frames} frames")
    st.info("Processing every 5th frame...")

    frame_data = []
    class_counter = Counter()
    frame_class_diversity = {}
    frame_id = 0
    processed_id = 0

    progress = st.progress(0)
    total_to_process = total_frames // 5

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % 5 == 0:
            results = model(frame)
            detections = results.pandas().xyxy[0]
            frame_summary = []
            unique_classes = set()

            for _, row in detections.iterrows():
                label = row['name']
                bbox = [int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])]
                conf = float(row['confidence'])

                frame_summary.append({
                    "label": label,
                    "bbox": bbox,
                    "confidence": round(conf, 3)
                })

                class_counter[label] += 1
                unique_classes.add(label)

                # Annotate
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (bbox[0], bbox[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            frame_class_diversity[processed_id] = len(unique_classes)

            # Save frame
            out_path = f"output/annotated_frames/annotated_{processed_id:04d}.jpg"
            cv2.imwrite(out_path, frame)

            frame_data.append({
                "frame_id": processed_id,
                "detections": frame_summary
            })

            processed_id += 1
            progress.progress(min(processed_id / total_to_process, 1.0))

        frame_id += 1

    cap.release()
    progress.empty()

    # Save JSON
    json_path = "output/detection_summary.json"
    with open(json_path, 'w') as f:
        json.dump(frame_data, f, indent=4)

    # Plot chart
    st.subheader("📊 Object Frequency")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=list(class_counter.keys()), y=list(class_counter.values()), palette="tab10", ax=ax)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Most diverse frame
    max_div_frame = max(frame_class_diversity, key=frame_class_diversity.get)
    st.subheader("📸 Frame with Max Class Diversity")
    st.markdown(f"**Frame {max_div_frame}** with **{frame_class_diversity[max_div_frame]} unique classes**.")

    # Download outputs
    st.subheader("📥 Download Results")
    with open(json_path, "rb") as f:
        st.download_button("Download detection_summary.json", f, "detection_summary.json", "application/json")

    st.success("Processing Complete ✅")
