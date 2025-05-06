# services.py
import asyncio
from collections import defaultdict
from pydantic import BaseModel
from typing import List, Dict, Optional,Tuple,Any
import numpy as np
import cv2
import base64

from app.services.object_detector import detect_objects
from app.services.face_recognizer import recognize_faces
from app.services.caption_generator import generate_captions


# --- Data Models ---
class FrameData(BaseModel):
    frame_number: int
    timestamp: float
    image_b64: str

class FrameResult(BaseModel):
    frame_number: int
    timestamp: float
    annotated_image_b64: str
    caption: str

class BatchResponse(BaseModel):
    username: str
    camera_number: str
    batch_id: int
    results: List[FrameResult]

# --- Queues ---
input_queue = asyncio.Queue()
caption_queue = asyncio.Queue()

# --- Results Buffer ---
results_buffer: Dict[str, Dict[str, Dict[int, BatchResponse]]] = defaultdict(lambda: defaultdict(dict))



def get_numpy_frame(frame: str) -> np.ndarray:
    img_data = base64.b64decode(frame)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # This gives you a BGR image
    # if img is None:
    #     print("Image decoding failed. Skipping image")
    #     return np.zeros((360, 640, 3), dtype=np.uint8)  # Replace with blank image to prevent issues
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img # Convert BGR to RGB

def numpy_to_base64(image: np.ndarray) -> str:
    # Convert RGB to BGR for OpenCV
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Encode image as JPEG (use .jpg if preferred)
    success, encoded_image = cv2.imencode('.jpg', bgr_image)
    if not success:
        print("Image encoding failed. Skipping image")
    base64_str = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
    return base64_str

def decode_recognition(recog_results: List[Dict[str, Any]])-> Tuple[List[np.array],List[List[str]]]:
    annotated_frames ,persons= [],[]
    for result in recog_results:
        annotated_frames.append(result["frame"])
        persons_in_frame=[]
        for person_dict in result["recognized_faces"]:
            persons_in_frame.append(person_dict.get('name',"Unknown"))
        persons.append(persons_in_frame)
    return annotated_frames,persons


def run_detection_and_recognition(frames: List[FrameData]) -> Dict[int, str]:
    np_frames = [get_numpy_frame(frame_data.image_b64) for frame_data in frames]
    detection_results = detect_objects(frames = np_frames)
    recog_results = recognize_faces(frames=np_frames,input_format="RGB")
    annotated_frames,persons = decode_recognition(recog_results)
    return detection_results,annotated_frames,persons

def run_caption_generation(frames: List[FrameData], detection_results: List[Dict[str,int]],persons:List[List[str]]) -> Dict[int, str]:
    np_frames = [get_numpy_frame(frame_data.image_b64) for frame_data in frames]
    timestamps = [frame_data.timestamp for frame_data in frames]

    return generate_captions(np_frames,timestamps,detection_results,persons)




# --- Pipeline Worker Functions ---
async def detection_recognition_worker():
    while True:
        username, camera_number, batch_id, frames = await input_queue.get()
        detection_results,annotated_frames,persons = run_detection_and_recognition(frames)
        await caption_queue.put((username, camera_number, batch_id, frames, detection_results,annotated_frames,persons))
        input_queue.task_done()

async def caption_worker():
    while True:
        username, camera_number, batch_id, frames, detection_results,annotated_frames,persons = await caption_queue.get()
        captions = run_caption_generation(frames,detection_results,persons)

        results = [FrameResult(
            frame_number=f.frame_number,
            timestamp=f.timestamp,
            annotated_image_b64=numpy_to_base64(annotated_frames[f.frame_number]),
            caption=captions[f.frame_number]
        ) for f in frames]

        results_buffer[username][camera_number][batch_id] = BatchResponse(
            username=username,
            camera_number=camera_number,
            batch_id=batch_id,
            results=results
        )
        caption_queue.task_done()

# --- Service Interface Functions ---
async def enqueue_batch(username: str, camera_number: str, batch_id: int, frames: List[FrameData]):
    """Adds a new batch from a specific user/camera to the processing pipeline."""
    # Optional: Check if this batch ID is already processed or in buffer to prevent duplicates
    # if batch_id in results_buffer.get(username, {}).get(camera_number, {}):
    #     print(f"Warning: Batch {batch_id} for {username}/{camera_number} already exists in buffer. Skipping enqueue.")
    #     # Depending on desired behavior, you might raise an error or just ignore
    #     return False # Indicate not enqueued

    await input_queue.put((username, camera_number, batch_id, frames))
    print(f"Batch {batch_id} for user {username}, camera {camera_number} enqueued.")
    return True # Indicate successfully enqueued

def get_batch_result(username: str, camera_number: str, batch_id: int) -> Optional[BatchResponse]:
    """Retrieves a processed batch result for a user/camera and removes it from the buffer."""
    user_cam_buffer = results_buffer.get(username, {}).get(camera_number, {})
    if batch_id in user_cam_buffer:
        print(f"Retrieving and removing batch {batch_id} for user {username}, cam {camera_number} from buffer.")
        return user_cam_buffer.pop(batch_id) # Retrieve and delete in one step
    else:
        print(f"Batch {batch_id} for user {username}, cam {camera_number} not found in buffer.")
        return None

def start_pipeline_workers(num_detection_workers: int = 1, num_caption_workers: int = 1):
    """Creates and starts the background worker tasks."""
    print(f"Starting {num_detection_workers} detection worker(s) and {num_caption_workers} caption worker(s)...")
    # Clear any stale state if restarting
    # Note: This is simple; real restarts might need more robust state handling
    while not input_queue.empty(): input_queue.get_nowait()
    while not caption_queue.empty(): caption_queue.get_nowait()
    # results_buffer.clear() # Decide if buffer should persist across restarts

    for _ in range(num_detection_workers):
        asyncio.create_task(detection_recognition_worker())
    for _ in range(num_caption_workers):
        asyncio.create_task(caption_worker())
    print("Pipeline workers started.")
