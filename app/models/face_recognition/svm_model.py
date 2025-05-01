import os
import pickle
import json
from sklearn.svm import SVC
from deepface import DeepFace
from app import config
from typing import Optional
import numpy as np

def train_svm_model(
    dataset_path: str = config.FaceRecogConfig.FACE_IMAGE_DIR, 
    model_path: str = config.FaceRecogConfig.SVM_MODEL_PATH, 
    label_path: str = config.FaceRecogConfig.SVM_LABELS_PATH,
    model_name: Optional[str] = None,
    detector_backend: Optional[str] = None,
    normalization: Optional[str] = None,
    align: Optional[bool] = None,
):
    """
    Train an SVM model for face recognition using DeepFace embeddings.
    The model and label mapping are saved to disk.
    dataset_path: str: Path to the dataset containing subfolders of images for each person.
    model_path: str: Path to save the trained SVM model.
    label_path: str: Path to save the label mapping.
    model_name: str: Model for face recognition. Default is "ArcFace".
    detector_backend: str: Face detection backend. Default is "yolov11n".
    normalization: str: Normalization method. Default is "ArcFace".
    align: bool: Whether to align faces before recognition. Default is False.
    confidence_threshold: float: Confidence threshold for face recognition. Default is 0.5.
    """
    # Check if the dataset path exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
    model_name = model_name or config.FaceRecogConfig.FACE_RECOGNITION_MODEL
    detector_backend = detector_backend or config.FaceRecogConfig.FACE_DETECTOR_BACKEND
    normalization = normalization or config.FaceRecogConfig.FACE_RECOGNITION_MODEL
    align = align if align is not None else config.FaceRecogConfig.ALIGN_FACES
    # Initialize lists to hold embeddings and labels   
    embeddings = []
    labels = []
    label_dict = {}
    current_label = 0

    # Loop through each person folder
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue

        if person_name not in label_dict:
            label_dict[person_name] = current_label
            current_label += 1
        person_label = label_dict[person_name]

        # Loop through images
        for filename in os.listdir(person_path):
            if not (filename.endswith(".jpg") or filename.endswith(".png")):
                continue
            img_path = os.path.join(person_path, filename)
            
            # Use DeepFace to extract the face embeddings
            try:
                embedding = DeepFace.represent(img_path,model_name=model_name,
                                    enforce_detection=False,
                                    detector_backend=detector_backend,
                                    align=align,
                                    normalization=normalization)[0]["embedding"] # only first face detected
                embeddings.append(embedding)
                labels.append(person_label)
            except Exception as e:
                print(f"Skipped {filename} due to error: {str(e)}")
                continue

    # Train the classifier
    embeddings = np.array(embeddings)
    recognizer = SVC(kernel="rbf", probability=True)
    recognizer.fit(embeddings, labels)

    # Save the trained model
    with open(model_path, "wb") as f:
        pickle.dump(recognizer, f)

    # Save label mapping
    with open(label_path, "w") as f:
        json.dump(label_dict, f)

    print("Training complete. Model and labels saved.")
