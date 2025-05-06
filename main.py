import uvicorn
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from pydantic import BaseModel
from typing import List

from app.services import pipeline
from app.services.pipeline import FrameData
from app.utils.batch_id_utils import BatchIDTracker
from app.config import BATCH_ID_JSON

tracker = BatchIDTracker(BATCH_ID_JSON)
app = FastAPI(title="Surveillance Backend")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print("Application startup: Initializing pipeline workers...")
    pipeline.start_pipeline_workers(num_detection_workers=1, num_caption_workers=1)
    yield
    # Shutdown logic
    print("Shutting down...")



# --- API Request Model ---
class BatchRequest(BaseModel):
    username: str
    camera_number: str
    frames: List[FrameData] # Use the FrameData model from pipeline

# --- API Endpoints ---
@app.post("/process_batch", status_code=202) # 202 Accepted is suitable here
async def process_batch_endpoint(batch_req: BatchRequest):
    """
    API endpoint to accept a batch of frames for processing.
    The batch is added to the pipeline queue for asynchronous processing.
    """
    
    try:
        batch_id = tracker.get_next_batch_id(batch_req.username, batch_req.camera_number)
        enqueued = await pipeline.enqueue_batch(
            username=batch_req.username,
            camera_number=batch_req.camera_number,
            batch_id=batch_id,
            frames=batch_req.frames
        )
        if not enqueued:
            # Handle cases where enqueueing failed (e.g., duplicate batch ID)
             return {"status": "ignored", "batch_id": batch_id, "message": "Batch ID possibly already processed or buffered."}

        return {"status": "accepted", "batch_id": batch_id, "message": "Batch added to processing queue."}
    except Exception as e:
        print(f"Error enqueuing batch {batch_id} for {batch_req.username}/{batch_req.camera_number}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during batch enqueueing.")

@app.get("/get_result/{username}/{camera_number}/{batch_id}", response_model=Optional[BatchResponse])
async def get_result_endpoint(username: str, camera_number: str, batch_id: int):
    """
    API endpoint for the frontend to retrieve the results of a specific processed batch.
    If found, the result is returned and removed from the server buffer.
    """
    result = pipeline.get_batch_result(username, camera_number, batch_id)
    if result is None:
        # Result not found (either not processed yet, already retrieved, or invalid ID)
        raise HTTPException(status_code=404, detail=f"Result for Batch ID {batch_id} (User: {username}, Cam: {camera_number}) not found.")
    return result # FastAPI will automatically serialize the BatchResponse model


# --- Main execution ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)