import streamlit as st
import json
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt

# Try OpenCV import with fallback
try:
    import cv2
    VIDEO_BACKEND = 'opencv'
except ImportError as e:
    st.warning(f"OpenCV not available: {e}. Using imageio fallback.")
    try:
        import imageio
        from PIL import Image
        VIDEO_BACKEND = 'imageio'
    except ImportError:
        st.error("Neither OpenCV nor imageio available. Please install at least one.")
        st.stop()

st.title("🎯 Detection Summary Engine")

class DetectionSummaryEngine:
    def __init__(self, model_type='yolov5'):
        self.model = self.load_model(model_type)
        self.results = {
            'metadata': {},
            'frames': {},
            'analytics': {'class_counts': defaultdict(int)}
        }
    
    def load_model(self, model_type):
        # Your model loading code here
        pass
    
    def process_video(self, video_path, every_n=5):
        if VIDEO_BACKEND == 'opencv':
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Rest of OpenCV processing
        else:
            reader = imageio.get_reader(video_path)
            fps = reader.get_meta_data()['fps']
            total_frames = reader.count_frames()
          
        self.results['metadata'].update({
            'video_file': video_path,
            'total_frames': total_frames,
            'processed_frames': total_frames // every_n,
            'fps': fps
        })
        
        max_diversity = 0
        max_diversity_frame = None
        
        for i in range(0, total_frames, every_n):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame_key = f"frame_{i:03d}"
            detections = self.detect_objects(frame)
            self.results['frames'][frame_key] = detections
            
            # Update analytics
            unique_classes = set()
            for obj in detections['objects']:
                cls_name = obj['class']
                self.results['analytics']['class_counts'][cls_name] += 1
                unique_classes.add(cls_name)
                
            if len(unique_classes) > max_diversity:
                max_diversity = len(unique_classes)
                max_diversity_frame = frame_key
                
        self.results['analytics']['max_diversity_frame'] = max_diversity_frame
        self.results['analytics']['max_diversity_count'] = max_diversity
        cap.release()
        
    def visualize_counts(self):
        counts = self.results['analytics']['class_counts']
        plt.figure(figsize=(10, 6))
        plt.bar(counts.keys(), counts.values())
        plt.title('Object Detection Summary')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('detection_summary.png')
        plt.show()
        
    def save_results(self, output_path):
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
