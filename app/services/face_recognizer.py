import cv2
import pickle
import json
import numpy as np
from deepface import DeepFace
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from app.models.face_recognition.svm_model import SVMFaceRecognizer
from app.models.face_recognition.Embedder import FaceEmbeddingManager
from app.config import FaceRecogConfig, get_dynamic_config


def recognize_faces(
        frames: List[np.ndarray], 
        input_format: str = "RGB",
        model_name: Optional[str] = None,
        detector_backend: Optional[str] = None,
        normalization: Optional[str] = None,
        align: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
    """
    Recognize faces in the given frames using DeepFace.
    
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
    model_name = model_name or FaceRecogConfig.FACE_RECOGNITION_MODEL
    detector_backend = detector_backend or FaceRecogConfig.FACE_DETECTOR_BACKEND
    normalization = normalization or FaceRecogConfig.FACE_RECOGNITION_MODEL
    align = align if align is not None else FaceRecogConfig.ALIGN_FACES
    
    if FaceRecogConfig.RECOGNIZER == "SVM":
        return recognize_faces_svm(frames, input_format, model_name, detector_backend, normalization, align)
    elif FaceRecogConfig.RECOGNIZER == "COSINE":
        return recognize_faces_cosine(frames, input_format, model_name, detector_backend, normalization, align)
    else:
        raise ValueError(f"Unknown recognizer: {FaceRecogConfig.RECOGNIZER}. Supported: SVM, COSINE")


svm_model = None
svm_label_encoder = None
svm_conf_threshold = 0.5

def load_svm_model(model_path:Union[str,Path] = FaceRecogConfig.SVM_MODEL_PATH, 
                   label_path:Union[str,Path] = FaceRecogConfig.SVM_LABELS_PATH) -> Tuple[SVC, LabelEncoder]:
    """
    Load the SVM model and label dictionary from the specified paths.
    """

    global svm_model, svm_label_encoder
    if svm_model is not None and  svm_label_encoder is not None:
        return svm_model,  svm_label_encoder,svm_conf_threshold
    # Load the trained SVM model
    svm_recognizer =SVMFaceRecognizer(model_path=model_path, label_encoder_path=label_path)
    svm_model = svm_recognizer.svm_model
    svm_label_encoder = svm_recognizer.label_encoder
    svm_conf_threshold = svm_recognizer.best_threshold

    if svm_model is None or svm_label_encoder is None:
        raise ValueError("SVM model or label encoder not found. Please train the model first.")

   
    return svm_model, svm_label_encoder,svm_conf_threshold

def recognize_faces_svm(
        frames: List[np.ndarray], 
        input_format: str = "RGB",
        model_name: Optional[str] = None,
        detector_backend: Optional[str] = None,
        normalization: Optional[str] = None,
        align: Optional[bool] = None,
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

    model_name = model_name or FaceRecogConfig.FACE_RECOGNITION_MODEL
    detector_backend = detector_backend or FaceRecogConfig.FACE_DETECTOR_BACKEND
    normalization = normalization or FaceRecogConfig.FACE_RECOGNITION_MODEL
    align = align if align is not None else FaceRecogConfig.ALIGN_FACES
    

    if input_format == "RGB":
        frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]
    # Use DeepFace to detect and get embeddings
    svm_model, svm_label_encoder ,confidence_threshold = load_svm_model()
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

        if not frame_detections:
            results.append({
                "frame": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                "recognized_faces": []
            })
            continue

        # Extract embeddings for all faces in the current frame
        embeddings = [face["embedding"] for face in frame_detections]
        regions = [face["facial_area"] for face in frame_detections]

        # Predict labels for all embeddings in this frame
        probs_batch = svm_model.predict_proba(embeddings)

        for face, region, probs in zip(frame_detections, regions, probs_batch):
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            best_idx = np.argmax(probs)
            confidence = probs[best_idx]
            name = svm_label_encoder.inverse_transform([best_idx])[0] if confidence >= confidence_threshold else "Unknown"

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({confidence*100:.1f}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            recognized_faces.append({"name": name, "confidence": confidence})

        results.append({
            "frame": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            "recognized_faces": recognized_faces
        })

    
    return results

Embeddings=None
COSINE_CONF_THRESHOLD = get_dynamic_config().get("BEST_COSINE_THRESHOLD", 0.5)

def load_embeddings(embeddings_path: Union[str, Path] = FaceRecogConfig.EMBEDDINGS_PATH) -> Dict[str, Any]:
    """
    Load the embeddings from the specified path.
    Args:
        embeddings_path (Union[str, Path]): Path to the embeddings file.
    Returns:
        Dict[str, Any]: Dictionary containing the embeddings.
    """
    if not Path(embeddings_path).exists():
        raise FileNotFoundError(f"Embeddings file not found at {embeddings_path}.")
    global Embeddings
    if Embeddings is not None:
        return Embeddings
    embedding_manager = FaceEmbeddingManager(embeddings_path=embeddings_path)
    _ = embedding_manager.load_embeddings()
    embeddings = embedding_manager.get_flatten_embeddings()
    if embeddings is None:
        raise ValueError("Failed to load embeddings. Please check the file format.")
    return embeddings


def recognize_faces_cosine(
        frames: List[np.ndarray],
        input_format: str = "RGB",
        model_name: Optional[str] = None,
        detector_backend: Optional[str] = None,
        normalization: Optional[str] = None,
        align: Optional[bool] = None,
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
    global COSINE_CONF_THRESHOLD
    model_name = model_name or FaceRecogConfig.FACE_RECOGNITION_MODEL
    detector_backend = detector_backend or FaceRecogConfig.FACE_DETECTOR_BACKEND
    normalization = normalization or FaceRecogConfig.FACE_RECOGNITION_MODEL
    align = align if align is not None else FaceRecogConfig.ALIGN_FACES
    
    if input_format == "RGB":
        frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in frames]
    # Use DeepFace to detect and get embeddings
    
    detections = DeepFace.represent(frames, model_name=model_name,
                                    enforce_detection=False,
                                    detector_backend=detector_backend,
                                    align=align,
                                    normalization=normalization)
    
    assert len(detections) == len(frames), "Number of frames and detections do not match"
    
    results = []
    embeddings = load_embeddings()
    for idx, frame_detections in enumerate(detections):
        frame = frames[idx]
        recognized_faces = []

        if not frame_detections:
            results.append({
                "frame": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                "recognized_faces": []
            })
            continue

        # Extract embeddings for all faces in the current frame
        embeddings_batch = [face["embedding"] for face in frame_detections]
        regions = [face["facial_area"] for face in frame_detections]

        # Calculate cosine similarity with the loaded embeddings
        for face, region, embedding in zip(frame_detections, regions, embeddings_batch):
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            best_idx = None
            best_score = -1.0

            for idx, stored_embedding in enumerate(embeddings):
                score = cosine_similarity([embedding], [stored_embedding])[0][0]
                if score > best_score:
                    best_score = score
                    best_idx = idx

            confidence = best_score
            name = str(best_idx) if confidence >= COSINE_CONF_THRESHOLD else "Unknown"

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({confidence*100:.1f}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            recognized_faces.append({"name": name, "confidence": confidence})

        results.append({
            "frame": cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
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

    model_name = model_name or FaceRecogConfig.FACE_RECOGNITION_MODEL
    detector_backend = detector_backend or FaceRecogConfig.FACE_DETECTOR_BACKEND
    normalization = normalization or FaceRecogConfig.FACE_RECOGNITION_MODEL
    align = align if align is not None else FaceRecogConfig.ALIGN_FACES
    confidence_threshold = confidence_threshold or get_dynamic_config().get("BEST_COSINE_THRESHOLD", 0.5)

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
