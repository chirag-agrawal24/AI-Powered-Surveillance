import os
import json
import numpy as np
from typing import Optional, Union
from pathlib import Path

from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import cosine_similarity

from app.models.face_recognition.Embedder import FaceEmbeddingManager 
from app.config import COSINEConfig,  get_dynamic_config, get_config_json_path
def tune_cosine_threshold(
        model_name:str="ArcFace",
        detector_backend:str="yolov8",
        emb_path:Union[str,Path]=COSINEConfig.EMBEDDING_PATH,
        dataset_path:Optional[Union[str,Path]]=None,
        force:bool=False,
        plot:bool=False
) -> float:
    """
    Tune the cosine similarity threshold for face recognition.
    """
    if dataset_path is None and emb_path is None:
        raise ValueError("Either dataset_path or emb_path must be provided.")
    embedding_manager = FaceEmbeddingManager(dataset_path=dataset_path,
                                                 model_name=model_name,
                                                 detector_backend=detector_backend,
                                                 emb_path=emb_path)
    # Load embeddings if not already in memory
    if os.path.exists(emb_path) and not force:
        embedding_store = embedding_manager.load_embeddings()
    else:
        embedding_store = embedding_manager.update_embeddings(force=force)


    same_scores = []
    diff_scores = []

    people = list(embedding_store.keys())
    
    # Positive pairs (same person)
    for person in people:
        embeddings = embedding_store[person]
        for img_name1, embeddings1 in embeddings.items():
            for img_name2, embeddings2 in embeddings.items():
                if img_name1 == img_name2:
                    continue
                sim = cosine_similarity([embeddings1], [embeddings2])[0][0]
                same_scores.append(sim)

    # Negative pairs (different people)
    for i in range(len(people)):
        for j in range(i+1, len(people)):
            emb1 = embedding_store[people[i]]
            emb2 = embedding_store[people[j]]
            for img_name1, e1 in emb1.items():
                for img_name2, e2 in emb2.items():
                    sim = cosine_similarity([e1], [e2])[0][0]
                    diff_scores.append(sim)

    # Labels: 1 = same, 0 = different
    labels = [1] * len(same_scores) + [0] * len(diff_scores)
    scores = same_scores + diff_scores

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    dynamic_config = get_dynamic_config()
    dynamic_config["BEST_COSINE_THRESHOLD"] = optimal_threshold
    with open(get_config_json_path(), "w") as f:
        json.dump(dynamic_config, f, indent=4)

    
    if plot:
        import matplotlib.pyplot as plt
        # Plot ROC Curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label=f'Best threshold = {optimal_threshold:.3f}')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()

        # Optional: Histogram too
        plt.hist(same_scores, bins=30, alpha=0.6, label="Same Person")
        plt.hist(diff_scores, bins=30, alpha=0.6, label="Different Person")
        plt.axvline(optimal_threshold, color='r', linestyle='--', label=f"Threshold = {optimal_threshold:.2f}")
        plt.title("Cosine Similarity Distribution")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid()
        plt.show()

    print(f"Suggested threshold: {optimal_threshold:.3f}")
    return optimal_threshold

    
if __name__ == "__main__":
    # Example usage
    tune_cosine_threshold(
        model_name="ArcFace",
        detector_backend="yolov8",
        emb_path=COSINEConfig.EMBEDDING_PATH,
        dataset_path=COSINEConfig.DATESET_PATH,
        force=True,
        plot=True
    )