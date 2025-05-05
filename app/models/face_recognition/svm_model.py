import os
import json
import numpy as np
import pickle
from typing import Optional, Union
from pathlib import Path

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from app.models.face_recognition.Embedder import FaceEmbeddingManager 
from app.config import SVMConfig, get_dynamic_config, get_config_json_path

# Function to train the SVM model
class SVMFaceRecognizer:
    def __init__(self,
        svm_model_path:Union[str,Path]=SVMConfig.SVM_MODEL_PATH, 
        label_encoder_path:Union[str,Path]=SVMConfig.SVM_LABEL_ENCODER_PATH
    ):
        """
        Initialize the SVMFaceRecognizer.
        svm_model_path: Path to save/load the SVM model.
        label_encoder_path: Path to save/load the label encoder.
        """
        self.svm_model_path=svm_model_path
        self.label_encoder_path = label_encoder_path
        if os.path.exists(self.svm_model_path) and os.path.exists(self.label_encoder_path):
            self.svm_model, self.label_encoder = self.load_svm_model()
        else:
            self.svm_model = None
            self.label_encoder = None
        self.best_threshold = get_dynamic_config().get("BEST_SVM_THRESHOLD", 0.5)

    def train_svm_model(self, 
        model_name:str="ArcFace",
        detector_backend:str="yolov8",
        emb_path:Union[str,Path]=SVMConfig.EMBEDDING_PATH,
        dataset_path:Optional[Union[str,Path]]=None,
        force:bool=False
    ):
        """
        Either give dataset_path or path to already generated embeddings.pkl file
        If pickle is not present or force=True then generate embedding from dataset
        and save it to emb_path.

        dataset_path: Path to the dataset folder containing subfolders with images of each person.
        model_name: Name of the DeepFace model to use for embedding generation.
        detector_backend: Backend for face detection (e.g., 'opencv', 'ssd', 'mtcnn', etc.).
        emb_path: Path to save/load the embeddings.
        force: If True, overwrite existing embeddings even if they are already present.
        """

        if dataset_path is None and emb_path is None:
            raise ValueError("Either dataset_path or emb_path must be provided.")
        embedding_manager = FaceEmbeddingManager(dataset_path=dataset_path,
                                                 model_name=model_name,
                                                 detector_backend=detector_backend,
                                                 emb_path=emb_path)
        # Load embeddings if not already in memory
        if os.path.exists(emb_path) and not force:
            _ = embedding_manager.load_embeddings()
        else:
            _ = embedding_manager.update_embeddings(force=force)
    
        X, y = embedding_manager.get_flatten_embeddings()
        X = np.array(X)
        # Label encoding
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
    
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
        # SVM model training
        self.svm_model = SVC(C=SVMConfig.SVM_C,
                        degree=SVMConfig.SVM_DEGREE,
                        gamma=SVMConfig.SVM_GAMMA,
                        kernel=SVMConfig.SVM_KERNEL,
                        probability=True)
        self.svm_model.fit(X_train, y_train)

        self.save_svm_model()
        self._tune_threshold_for_svm(X_train, y_train)

        y_prob = self.svm_model.predict_proba(X_test)
        y_pred = np.argmax(y_prob, axis=1)

        # Apply thresholding to probabilities
        mask = np.max(y_prob, axis=1) >= self.best_threshold
        y_pred_thresholded = np.where(mask, y_pred, -1)
        

        accuracy = accuracy_score(y_test, y_pred_thresholded)
        precision = precision_score(y_test, y_pred_thresholded, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred_thresholded, average='weighted', zero_division=0)

        print(f"SVM Model Performance (with threshold {self.best_threshold:.2f}):")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        return self.svm_model, self.label_encoder

    
    # Function to load the trained SVM model and label encoder
    def load_svm_model(self):
        """
        Load the SVM model and label encoder from the specified paths.
        """
        with open(self.svm_model_path, "rb") as f:
            self.svm_model = pickle.load(f)
    
        with open(self.label_encoder_path, "rb") as f:
            self.label_encoder = pickle.load(f)
    
        return self.svm_model, self.label_encoder
    def save_svm_model(self):
        """
        Save the SVM model and label encoder to the specified paths.
        """
        with open(self.svm_model_path, "wb") as f:
            pickle.dump(self.svm_model,f)
    
        with open(self.label_encoder_path, "wb") as f:
            pickle.dump(self.label_encoder,f)
    
    # Function to tune confidence threshold
    def _tune_threshold_for_svm(self,X_train, y_train):

        y_prob_train = self.svm_model.predict_proba(X_train)
    
        best_f1 = 0
        thresholds = np.linspace(0, 1, 100)
        
        for threshold in thresholds:
            # Compute predicted class only if max prob ≥ threshold; else assign -1
            mask = np.max(y_prob_train, axis=1) >= threshold
            y_train_pred = np.where(mask, np.argmax(y_prob_train, axis=1), -1)

            
            

            f1_train = f1_score(y_train, y_train_pred, average='weighted')

            if f1_train > best_f1:
                best_f1 = f1_train
                self.best_threshold = threshold

        
        dynamic_config = get_dynamic_config()
        dynamic_config["BEST_SVM_THRESHOLD"] = self.best_threshold
        with open(get_config_json_path(), "w") as f:
            json.dump(dynamic_config, f, indent=4)
    
        print(f"Best Threshold: {self.best_threshold:.2f}, Best F1-Score: {best_f1:.4f}")
        return self.best_threshold
    


if __name__ == "__main__":
    # Example usage
    svm_recognizer = SVMFaceRecognizer()
    print("Training SVM model on dataset:", SVMConfig.DATESET_PATH)
    
    svm_recognizer.train_svm_model(dataset_path=SVMConfig.DATESET_PATH,force=False)
    # Load the model later
    svm_recognizer.load_svm_model()














