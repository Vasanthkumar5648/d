import cv2
import torch
import json
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Load YOLOv5s model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

# Paths
VIDEO_PATH = 'input_video.mp4'
OUTPUT_DIR = 'output'
FRAME_DIR = os.path.join(OUTPUT_DIR, 'frames')
ANNOTATED_DIR = os.path.join(OUTPUT_DIR, 'annotated_frames')
os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

print(f"Video duration: {duration:.2f}s | Total frames: {total_frames} | FPS: {fps}")

frame_data = []
class_counter = Counter()
frame_class_diversity = {}

frame_id = 0
processed_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % 5 == 0:
        raw_path = os.path.join(FRAME_DIR, f"frame_{processed_id:04d}.jpg")
        cv2.imwrite(raw_path, frame)

        results = model(frame)
        detections = results.pandas().xyxy[0]

        frame_summary = []
        classes_this_frame = set()

        for _, row in detections.iterrows():
            label = row['name']
            bbox = [int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])]
            confidence = float(row['confidence'])

            frame_summary.append({
                "label": label,
                "bbox": bbox,
                "confidence": round(confidence, 3)
            })

            class_counter[label] += 1
            classes_this_frame.add(label)

            # Annotate
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (bbox[0], bbox[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Save annotated frame
        ann_path = os.path.join(ANNOTATED_DIR, f"annotated_{processed_id:04d}.jpg")
        cv2.imwrite(ann_path, frame)

        frame_class_diversity[processed_id] = len(classes_this_frame)

        frame_data.append({
            "frame_id": processed_id,
            "detections": frame_summary
        })

        processed_id += 1

    frame_id += 1

cap.release()

# Save JSON
json_path = os.path.join(OUTPUT_DIR, 'detection_summary.json')
with open(json_path, 'w') as f:
    json.dump(frame_data, f, indent=4)

# Print object count
print("\n🔍 Object Count:")
for label, count in class_counter.items():
    print(f"{label}: {count}")

# Bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x=list(class_counter.keys()), y=list(class_counter.values()), palette="Set2")
plt.title("Object Frequency")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'object_frequency.png'))
plt.close()

# Most diverse frame
max_diverse_frame = max(frame_class_diversity, key=frame_class_diversity.get)
print(f"\n📸 Frame with max class diversity: Frame {max_diverse_frame} with {frame_class_diversity[max_diverse_frame]} classes")
print(f"\n✅ JSON saved to {json_path}")
