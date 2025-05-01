import os
from pathlib import Path
from dataclasses import dataclass
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings

@dataclass
class FaceRecogConfig:
    FACE_IMAGE_DIR: Path = (Path(__file__).parent.resolve() / "../data/input/face_images").resolve()
    FACE_RECOGNITION_MODELS_DIR: Path = Path(__file__).parent.resolve() / "models/face_recognition"
    SVM_MODEL_PATH: Path = FACE_RECOGNITION_MODELS_DIR / "svm_face_model.pkl"
    SVM_LABELS_PATH: Path = FACE_RECOGNITION_MODELS_DIR / "labels.json"
    FACE_CONFIDENCE_THRESHOLD: float = 0.5  # Adjust between 0-1 as needed
    FACE_DETECTOR_BACKEND: str = "yolov11n"  # Options: "opencv", "ssd", "dlib", "mtcnn", "retinaface", "yolo", "mediapipe"
    FACE_RECOGNITION_MODEL: str = "ArcFace"  # Options: "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace", "SFace", "GhostFaceNet"
    ALIGN_FACES: bool = False  # Whether to align faces before recognition
