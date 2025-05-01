import cv2
import pickle
import json
import numpy as np
from deepface import DeepFace
from app import config
from typing import Any, Dict, List, Tuple, Optional

svm_model = None
svm_label_dict = None

def load_svm_model(model_path:str = config.FaceRecogConfig.SVM_MODEL_PATH, 
                   label_path:str = config.FaceRecogConfig.SVM_LABELS_PATH) -> Tuple[Any, Dict[int, str]]:
    """
    Load the SVM model and label dictionary from the specified paths.
    """

    global svm_model, svm_label_dict
    if svm_model is not None and svm_label_dict is not None:
        return svm_model, svm_label_dict
    # Load the trained SVM model
    with open(model_path, "rb") as f:
        svm_model = pickle.load(f)

    # Load the label dictionary
    with open(label_path, "r") as f:
        label_dict = json.load(f)
    svm_label_dict = {v: k for k, v in label_dict.items()}
    return svm_model, svm_label_dict

def recognize_faces_svm(
        frames: List[np.ndarray], 
        input_format: str = "RGB",
        model_name: Optional[str] = None,
        detector_backend: Optional[str] = None,
        normalization: Optional[str] = None,
        align: Optional[bool] = None,
        confidence_threshold: Optional[float] = None
        ) -> List[Dict[str, Any]]:
    """
    Recognize faces in the given frames using DeepFace and SVM model.
    
    Args:
        frames (List[np.ndarray]): List of frames (images).
        input_format (str): Format of the input images(RGB or BGR). Default is "RGB".
        model_name (str): Model for face recognition. Default is "ArcFace".
        detector_backend (str): Face detection backend. Default is "yolov11n".
        normalization (str): Normalization method. Default is "ArcFace".
        align (bool): Whether to align faces before recognition. Default is False.

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing :
            
            - "frame": The processed frame with recognized faces in RGB format.
            - "recognized_faces": List of recognized faces with their names and confidence scores.
    """

    model_name = model_name or config.FaceRecogConfig.FACE_RECOGNITION_MODEL
    detector_backend = detector_backend or config.FaceRecogConfig.FACE_DETECTOR_BACKEND
    normalization = normalization or config.FaceRecogConfig.FACE_RECOGNITION_MODEL
    align = align if align is not None else config.FaceRecogConfig.ALIGN_FACES
    confidence_threshold = confidence_threshold or config.FaceRecogConfig.FACE_CONFIDENCE_THRESHOLD

    if input_format == "RGB":
        frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]
    # Use DeepFace to detect and get embeddings
    svm_model, svm_label_dict = load_svm_model(config.SVM_MODEL_PATH, config.SVM_LABELS_PATH)
    detections = DeepFace.represent(frames, model_name=model_name,
                                    enforce_detection=False,
                                    detector_backend=detector_backend,
                                    align=align,
                                    normalization=normalization)
    
    assert len(detections) == len(frames), "Number of frames and detections do not match"
    
    results = []
    for idx, frame_detections in enumerate(detections):
        frame = frames[idx]
        recognized_faces = []
        for face in frame_detections:
            embedding = face["embedding"]
            # Predict the label using the SVM recognizer
            region = face["facial_area"]  # dict with keys: x, y, w, h
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]

            probs = svm_model.predict_proba([embedding])[0]
            best_idx = np.argmax(probs)
            confidence = probs[best_idx]
            name = svm_label_dict.get(best_idx, "Unknown") if confidence >= confidence_threshold else "Unknown"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({confidence*100:.1f}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            recognized_faces.append({"name": name, "confidence": confidence})
        results.append({
            "frame": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),  # Convert BGR to RGB for display
            "recognized_faces": recognized_faces
        })
    
    return results

def recognize_faces_vectordb(
        frames: List[np.ndarray], 
        vector_db,  # e.g., FAISS index or wrapper
        label_lookup: Dict[int, str],
        input_format: str = "RGB",
        model_name: Optional[str] = None,
        detector_backend: Optional[str] = None,
        normalization: Optional[str] = None,
        align: Optional[bool] = None,
        confidence_threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Recognize faces in the given frames using a vector database for face recognition.
    Args:
        frames (List[np.ndarray]): List of frames (images).
        input_format (str): Format of the input images(RGB or BGR). Default is "RGB".
        model_name (str): Model for face recognition. Default is "ArcFace".
        detector_backend (str): Face detection backend. Default is "yolov11n".
        normalization (str): Normalization method. Default is "ArcFace".
        align (bool): Whether to align faces before recognition. Default is False.

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing :
            
            - "frame": The processed frame with recognized faces in RGB format.
            - "recognized_faces": List of recognized faces with their names and confidence scores.
    """

    model_name = model_name or config.FaceRecogConfig.FACE_RECOGNITION_MODEL
    detector_backend = detector_backend or config.FaceRecogConfig.FACE_DETECTOR_BACKEND
    normalization = normalization or config.FaceRecogConfig.FACE_RECOGNITION_MODEL
    align = align if align is not None else config.FaceRecogConfig.ALIGN_FACES
    confidence_threshold = confidence_threshold or config.FaceRecogConfig.FACE_CONFIDENCE_THRESHOLD

    if input_format == "RGB":
        frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]

    detections = DeepFace.represent(frames, model_name=model_name,
                                    enforce_detection=False,
                                    detector_backend=detector_backend,
                                    align=align,
                                    normalization=normalization)

    results = []
    for idx, frame_detections in enumerate(detections):
        frame = frames[idx]
        recognized_faces = []
        for face in frame_detections:
            embedding = face["embedding"]
            region = face["facial_area"]
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]

            distances, indices = vector_db.search(np.array([embedding]), k=1)  # FAISS example
            best_idx = indices[0][0]
            distance = distances[0][0]
            confidence = 1 / (1 + distance)  # Optional scoring function
            name = label_lookup.get(best_idx, "Unknown") if confidence >= confidence_threshold else "Unknown"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"{name} ({confidence*100:.1f}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            recognized_faces.append({"name": name, "confidence": confidence})

        results.append({
            "frame": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            "recognized_faces": recognized_faces
        })
    return results
