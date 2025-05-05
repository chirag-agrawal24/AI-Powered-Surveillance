import logging
from collections import Counter
from ultralytics import YOLO
import torch
import os

from typing import List
import numpy as np


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from app.config import ObjectDetectionConfig

device = ObjectDetectionConfig.DEVICE  # Get the device from the configuration
if device !="cuda":
    logger.info("Using {device} for Object Detection inference.")
    pass
elif torch.cuda.is_available() and device == "cuda":
    device = "cuda"
    logger.info("Using GPU for Object Detection inference.")
else:
    device = "cpu"
    logger.info("Using CPU for Object Detection inference.")

if os.path.exists(ObjectDetectionConfig.MODEL_PATH):
    model = YOLO(ObjectDetectionConfig.MODEL_PATH).to(device)
    logger.info(f"Loaded model from {ObjectDetectionConfig.MODEL_PATH}")
else:
    model = YOLO("yolo11n.pt").to(device)
    logger.info("Loaded default yolo11n.pt")

def detect_objects(frames:List[np.array])-> list:
    """
    Detect objects in a list of frames using YOLOv8 model.
    Args:
        frames (list): List of frames to process.
    Returns:
        list: List of detected objects in each frame.
    """
    try:
        results = model.predict(frames, verbose=False,)
        detected_objects = []
        for frame_idx, result in enumerate(results):
            names = [result.names[cls.item()] for cls in result.boxes.cls.int()]
            count = Counter(names)
            detected_objects.append({
                "frame_num": frame_idx,
                "objects": count
            })
        return detected_objects
    except Exception as e:
        logger.error(f"Error in object detection: {str(e)}")
        return []
if __name__ == "__main__":
    import cv2
    test_path = ObjectDetectionConfig.TEST_PATH  # Path to test images
    frames = []
    for file in os.listdir(test_path):
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
            image_path = os.path.join(test_path, file)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)
            frames.append(image)
    result = detect_objects(frames)  # Example usage with a dummy frame
    for res in result:
        print(f"Frame {res['frame_num']}:")
        for obj, count in res['objects'].items():
            print(f"  {obj}: {count}")
        print()
    print("Object detection completed.")