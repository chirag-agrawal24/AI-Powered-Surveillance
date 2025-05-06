# # Version 1
# from typing import List,Dict
# import numpy as np

# def generate_captions(frames:List[np.array], timestamps,detection_results: List[Dict[str,int]],persons:List[List[str]]):
#     return ["Dummy caption" for frame in frames]

from typing import List, Dict
import numpy as np
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import difflib
from pydantic import BaseModel
from supabase import create_client, Client
from datetime import datetime
import pytz
from collections import Counter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
import gc

# Initialize Supabase client
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda").eval()

# Initialize Google Generative AI Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# Global variable to track the last printed caption
last_caption = None

class FrameData(BaseModel):
    timestamp: float
    caption_text: str
    objects: Dict
    persons: List[str]

def generate_captions(frames: List[np.ndarray], timestamps: List[float], detection_results: List[Dict[str, int]], persons: List[List[Dict]]):
    global last_caption
    results = []
    pil_frames = [Image.fromarray(frame).convert('RGB') for frame in frames]
    conditional_prompt = (
        "Describe the actions of the recognized person in this CCTV footage. "
        "Use the person's name in the description if provided, and include relevant objects or context."
    )

    # Chunking parameters
    MAX_CHUNK_LENGTH = 500
    TIME_WINDOW = 5000
    chunk_data = {
        "start_timestamp": None,
        "end_timestamp": None,
        "objects": Counter(),
        "persons": set(),
        "captions": []
    }

    def save_chunk():
        if chunk_data["captions"]:
            caption_text = "; ".join(chunk_data["captions"])
            try:
                embedding = embeddings.embed_query(caption_text)
                supabase.table("caption_chunks").insert({
                    "start_timestamp": datetime.fromtimestamp(chunk_data["start_timestamp"] / 1000, tz=pytz.UTC).isoformat(),
                    "end_timestamp": datetime.fromtimestamp(chunk_data["end_timestamp"] / 1000, tz=pytz.UTC).isoformat(),
                    "objects": dict(chunk_data["objects"]),
                    "persons": list(chunk_data["persons"]),
                    "caption_text": caption_text,
                    "caption_embedding": embedding
                }).execute()
            except Exception as e:
                print(f"Error generating embedding or inserting chunk: {e}")
            chunk_data["start_timestamp"] = None
            chunk_data["end_timestamp"] = None
            chunk_data["objects"].clear()
            chunk_data["persons"].clear()
            chunk_data["captions"].clear()

    for idx, (frame, timestamp) in enumerate(zip(pil_frames, timestamps)):
        try:
            frame_results = {
                "timestamp": timestamp,
                "captions": []
            }

            detection_data = detection_results[idx] if idx < len(detection_results) else {}
            detected_persons = persons[idx] if idx < len(persons) else []

            detected_objects = detection_data.get("objects", {}) if isinstance(detection_data, dict) else {}
            objects_str = ", ".join([f"{count} {label}" for label, count in detected_objects.items()]) if detected_objects else "no objects detected"
            persons_str = ", ".join([person.get("name", "unknown") for person in detected_persons]) if detected_persons else "no persons recognized"
            persons_list = [person.get("name", "unknown") for person in detected_persons] if detected_persons else []

            if persons_str != "no persons recognized":
                enhanced_prompt = (
                    f"{conditional_prompt} The recognized person is {persons_str}. "
                    f"Relevant objects in the scene: {objects_str}."
                )
            else:
                enhanced_prompt = (
                    f"{conditional_prompt} No person is recognized. "
                    f"Relevant objects in the scene: {objects_str}."
                )

            inputs = processor(frame, enhanced_prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_length=100,
                    num_beams=7,
                    temperature=0.6,
                    top_p=0.8
                )
            caption = processor.decode(out[0], skip_special_tokens=True).strip()
            if caption.lower().startswith(conditional_prompt.lower()):
                caption = caption[len(conditional_prompt):].strip()
            elif caption.lower().startswith("the recognized person is"):
                caption = caption[caption.lower().find(persons_str.lower()):].strip()
            if persons_str != "no persons recognized":
                caption = caption.replace("a man", persons_str).replace("A man", persons_str)
            frame_results["captions"].append(caption)

            inputs = processor(frame, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_length=50,
                    num_beams=7,
                    temperature=0.7,
                    top_p=0.8
                )
            unconditional_caption = processor.decode(out[0], skip_special_tokens=True).strip()
            if persons_str != "no persons recognized":
                unconditional_caption = unconditional_caption.replace("a man", persons_str).replace("A man", persons_str)
            frame_results["captions"].append(unconditional_caption)

            if not chunk_data["start_timestamp"]:
                chunk_data["start_timestamp"] = timestamp
            chunk_data["end_timestamp"] = timestamp
            chunk_data["objects"].update(detected_objects)
            chunk_data["persons"].update(persons_list)
            chunk_data["captions"].extend(frame_results["captions"])

            chunk_text = "; ".join(chunk_data["captions"])
            if (len(chunk_text) > MAX_CHUNK_LENGTH or
                (chunk_data["end_timestamp"] - chunk_data["start_timestamp"]) > TIME_WINDOW):
                save_chunk()

            captions_to_print = []
            for caption in frame_results["captions"]:
                if last_caption is None or difflib.SequenceMatcher(None, caption, last_caption).ratio() < 0.9:
                    captions_to_print.append(caption)
                    last_caption = caption

            if captions_to_print:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp/1000))}]")
                print(f"Objects: {objects_str}")
                print(f"Persons: {persons_str}")
                for caption in captions_to_print:
                    print(caption)
                print()
            else:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp/1000))}] Skipped: No significant change in scene.")

            results.append(frame_results)

        except Exception as e:
            print(f"Error processing frame at timestamp {timestamp}: {e}")
            continue

    save_chunk()
    gc.collect()
    torch.cuda.empty_cache()
    return [result["captions"] for result in results]
