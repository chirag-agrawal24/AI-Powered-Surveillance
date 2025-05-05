import os
import pickle
from deepface import DeepFace
from pathlib import Path
from typing import Optional, Union


def flatten_embeddings(embedding_store):
    """
    Flatten the embedding store into a list of embeddings and their corresponding labels.
    """
    X, y = [], []
    for person, images in embedding_store.items():
        for _, embedding in images.items():
            X.append(embedding)
            y.append(person)
    return X, y

class FaceEmbeddingManager:
    """
    Class to manage face embeddings using DeepFace.
    """
    def __init__(self,
                 dataset_path:Union[str,Path]="data/input/", 
                 model_name:str="ArcFace", detector_backend:str="yolov8", 
                 emb_path:Union[str,Path]="embeddings.pkl"):
        """
        Initialize the FaceEmbeddingManager.
        dataset_path: Path to the dataset folder containing subfolders with images of each person.
        model_name: Name of the DeepFace model to use for embedding generation.
        detector_backend: Backend for face detection (e.g., 'opencv', 'ssd', 'mtcnn', etc.).
        emb_path: Path to save/load the embeddings.
        """
        
        self.dataset_path= dataset_path
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.emb_path = emb_path
        self.embedding_store = {}

    def _generate_embeddings_batch(self, image_paths):
        """
        Helper function to generate embeddings for a batch of images."""
        try:
            results = DeepFace.represent(
                img_path=image_paths,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                normalization=self.model_name,
                align=True,
                enforce_detection=False
            )
            return [res[0]["embedding"] for res in results] #Embedding of first face in image
        except Exception as e:
            print(f"[ERROR] Batch embedding failed: {e}")
            return []

    def update_embeddings(self,
                        dataset_path:Optional[Union[str,Path]]=None,
                        force:bool=False):
        """
        Append new images to the existing embeddings or create new ones if not present.
        dataset_path: Path to the dataset folder containing subfolders with images of each person.
        force: If True, overwrite existing embeddings even if they are already present.
        """
        if dataset_path is None:
            dataset_path = self.dataset_path
        if os.path.exists(self.emb_path) and not force:
            self.load_embeddings()
        else:
            self.embedding_store = {}
        print(f"[INFO] Loading images from {dataset_path}")
        
    
        for person in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person)
            if not os.path.isdir(person_path):
                continue

            if person not in self.embedding_store or force:
                self.embedding_store[person] = {}

            image_paths = []
            image_names = []
            for img_name in os.listdir(person_path):
                if not img_name.lower().endswith(("jpg", "jpeg", "png")):
                    continue
                if not force and img_name in self.embedding_store[person]:
                    continue
                image_paths.append(os.path.join(person_path, img_name))
                image_names.append(img_name)

            if image_paths:
                embeddings = self._generate_embeddings_batch(image_paths)
                for img_name, emb in zip(image_names, embeddings):
                    self.embedding_store[person][img_name] = emb

        self.save_embeddings()
        print("[INFO] Embeddings updated and saved.")
        return self.embedding_store

    def save_embeddings(self):
        """
        Save the embeddings to a file using pickle.
        """
        with open(self.emb_path, "wb") as f:
            pickle.dump(self.embedding_store, f)
        print(f"[INFO] Embeddings saved to {self.emb_path}")

    def load_embeddings(self):
        """
        Load the embeddings from a file using pickle.
        """
        with open(self.emb_path, "rb") as f:
            self.embedding_store = pickle.load(f)
        print(f"[INFO] Embeddings loaded from {self.emb_path}")
        return self.embedding_store
        
    def get_flatten_embeddings(self):
        """
        Get the flattened embeddings and their corresponding labels.
        """
        return flatten_embeddings(self.embedding_store)
        