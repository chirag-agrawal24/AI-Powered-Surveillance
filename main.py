import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi import UploadFile, File, Form, Request,Body
from fastapi.responses import JSONResponse,HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager

from pydantic import BaseModel
from typing import List,Optional,Union

from PIL import Image
import time
import numpy as np
import io
import asyncio
import json
import base64
from pathlib import Path

from app.services import pipeline
from app.services.pipeline import FrameData,BatchResponse
from app.utils.batch_id_utils import BatchIDTracker
from app.config import BATCH_ID_JSON , ServerConfig
from app.router import main_router




tracker = BatchIDTracker(BATCH_ID_JSON)
# --- API Request Model ---
class BatchRequest(BaseModel):
    username: str
    batch_id: Union[int,float, str]
    camera_number: str
    frames: List[FrameData] # Use the FrameData model from pipeline
class ResultRequest(BaseModel):
    username: str
    camera_number: str
    batch_id: int

def create_dummy_image_b64(width=640, height=360, color=(255, 0, 0)) -> str:
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    img_array[:, :] = color  # Fill with red, for example
    image = Image.fromarray(img_array)

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return encoded_image

# Warmup function
async def warmup():
    dummy_image_b64 = create_dummy_image_b64()

    frame = FrameData(
        frame_number=1,
        timestamp=time.time(),
        image_b64=dummy_image_b64
    )
    username = "dummy"
    batch_id = 1
    camera_number = "1"
    req = BatchRequest(
        username=username,
        batch_id=batch_id,
        camera_number=camera_number,
        frames=[frame for _ in range(5)]
    )
    result_req = ResultRequest(
        username=username,
        camera_number=camera_number,
        batch_id=batch_id
    )
    print("Warming up models...")
    _ = await process_batch_endpoint(req)
    start = time.time()
    while(time.time()-start < 15):
        try:
            _ = await get_result(result_req)
            print("Warm up Completed")
            break
        except Exception as e:
            pass
        await asyncio.sleep(5)
    else:
        print("Warm up stil running in background")

    


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print("Application startup: Initializing pipeline workers...")
    pipeline.start_pipeline_workers(num_detection_workers=1, num_caption_workers=1)
    await warmup()
    yield
    # Shutdown logic
    print("Shutting down...")

app = FastAPI(title="Surveillance Backend",lifespan=lifespan)
app.include_router(main_router)





# Mount static files directory
app.mount("/static", StaticFiles(directory=ServerConfig.STATIC_DIR), name="static")

# Jinja2 templates directory
templates = Jinja2Templates(directory=ServerConfig.TEMPLATES_DIR)


# ----------- ROUTES -----------

# Home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



@app.post("/api/get_result", response_model=Optional[BatchResponse])
async def get_result(data: ResultRequest = Body(...)):
    """
    API endpoint for the frontend to retrieve the results of a specific processed batch.
    If found, the result is returned and removed from the server buffer.
    """
    # print(data)
    result = pipeline.get_batch_result(data.username, data.camera_number, data.batch_id)
    if result is None:
        # Result not found (either not processed yet, already retrieved, or invalid ID)
        raise HTTPException(status_code=404, detail=f"Result for Batch ID {data.batch_id} (User: {data.username}, Cam: {data.camera_number}) not found.")
    return result # FastAPI will automatically serialize the BatchResponse model



# --- API Endpoints ---
@app.post("/api/process_batch", status_code=202) # 202 Accepted is suitable here
async def process_batch_endpoint(batch_req:BatchRequest = Body(...)):
    """
    API endpoint to accept a batch of frames for processing.
    The batch is added to the pipeline queue for asynchronous processing.
    """
    
    try:
        batch_id = batch_req.batch_id
        # batch_id = tracker.get_next_batch_id(batch_req.username, batch_req.camera_number)
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


@app.post("/api/process_batch_multipart")
async def process_batch_multipart(
    request: Request,
    username: str = Form(...),
    camera_number: str = Form(...),
    batch_id: int = Form(...),
    files: List[UploadFile] = File(...),
    frame_metadata: str = Form(...)
):
    try:
        metadata_list = json.loads(frame_metadata)
        metadata_map = {entry["filename"]: entry for entry in metadata_list}

        frames = []
        for file in files:
            content = await file.read()
            b64_str = base64.b64encode(content).decode("utf-8")
            meta = metadata_map.get(file.filename)

            if meta is None:
                return JSONResponse(status_code=400, content={"detail": f"Missing metadata for {file.filename}"})

            frames.append(FrameData(
                frame_number=meta["frame_number"],
                timestamp=meta["timestamp"],
                image_b64=b64_str
            ))

        enqueued = await pipeline.enqueue_batch(
            username=username,
            camera_number=camera_number,
            batch_id=batch_id,
            frames=frames
        )

        if not enqueued:
            return {"status": "ignored", "batch_id": batch_id, "message": "Batch ID possibly already processed or buffered."}

        return {"status": "accepted", "batch_id": batch_id, "message": "Batch added to processing queue."}

    except Exception as e:
        print(f"Error in multipart processing: {e}")
        return JSONResponse(status_code=500, content={"detail": "Internal server error during multipart batch enqueueing."})

    



# --- Main execution ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)