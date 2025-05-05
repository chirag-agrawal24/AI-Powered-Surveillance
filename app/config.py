import os
import json
from pathlib import Path
from dataclasses import dataclass

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings

# Load dynamic configuration from config.json

CONFIG_JSON_PATH = Path(__file__).parent.resolve() / "config.json"  
def get_config_json_path() -> Path:
    """Get the path to the config.json file."""
    return CONFIG_JSON_PATH
def get_dynamic_config() -> dict:
    """Get the dynamic configuration loaded from config.json."""
    if CONFIG_JSON_PATH.exists():
        with open(CONFIG_JSON_PATH, "r") as f:
            return json.load(f)
    else:
        return {}


@dataclass
class FaceRecogConfig:
    # Static constants
    FACE_IMAGE_DIR: Path = (Path(__file__).parent.resolve() / "../data/input/face_recog/face_images").resolve()
    FACE_RECOGNITION_MODELS_DIR: Path = Path(__file__).parent.resolve() / "models/face_recognition"
    SVM_MODEL_PATH: Path = FACE_RECOGNITION_MODELS_DIR / "svm_face_model.pkl"
    SVM_LABELS_PATH: Path = FACE_RECOGNITION_MODELS_DIR / "label_encoder.pkl"
    FACE_DETECTOR_BACKEND: str = "yolov11n"  # Options: "opencv", "ssd", "dlib", etc.
    FACE_RECOGNITION_MODEL: str = "ArcFace"  # Options: "VGG-Face", "Facenet", etc.
    ALIGN_FACES: bool = False  # Whether to align faces before recognition
    RECOGNIZER : str = "SVM"  # Options: "SVM", "COSINE"



@dataclass
class SVMConfig:
    # SVM specific configurations
    SVM_C: float = 1.0
    SVM_DEGREE: int = 2
    SVM_GAMMA: str = "scale"
    SVM_KERNEL: str = "rbf"
    SVM_MODEL_PATH: Path = FaceRecogConfig.SVM_MODEL_PATH
    SVM_LABEL_ENCODER_PATH: Path = FaceRecogConfig.SVM_LABELS_PATH
    EMBEDDING_PATH: Path = FaceRecogConfig.FACE_RECOGNITION_MODELS_DIR / "embeddings.pkl"
    DATESET_PATH: Path = FaceRecogConfig.FACE_IMAGE_DIR


@dataclass
class COSINEConfig:
    # Cosine specific configurations
    EMBEDDING_PATH: Path = FaceRecogConfig.FACE_RECOGNITION_MODELS_DIR / "embeddings.pkl"
    DATESET_PATH: Path = FaceRecogConfig.FACE_IMAGE_DIR
    
    DETECTOR_BACKEND: str = FaceRecogConfig.FACE_DETECTOR_BACKEND  
    FACE_RECOGNITION_MODEL: str = FaceRecogConfig.FACE_RECOGNITION_MODEL
    ALIGN_FACES: bool = FaceRecogConfig.ALIGN_FACES
    RECOGNIZER: str = FaceRecogConfig.RECOGNIZER
