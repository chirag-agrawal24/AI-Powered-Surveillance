
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


from app.config import ServerConfig
from app.api.logger import logger

live_cap_rounter = APIRouter()

# Process batch of frames
@live_cap_rounter.post("/api/process_batch")
async def process_batch(data: FrameBatch, background_tasks: BackgroundTasks):
    logger.info(f"Received batch processing request for camera: {data.camera_id}")
    try:
        camera_id = data.camera_id

        

        # Sort frames by timestamp to ensure chronological processing
        sorted_frames = sorted(data.frames, key=lambda x: x.timestamp)

        # Convert base64 frames to numpy arrays while preserving timestamps
        np_frames = []
        timestamps = []
        display_times = []

        for frame_data in sorted_frames:
            
            # Process the frame
            frame_b64 = frame_data.frame
            if ',' in frame_b64:
                frame_b64 = frame_b64.split(',')[1]

            # Update the latest timestamp
            latest_timestamp = frame_data.timestamp

            # Continue with normal processing
            img_data = base64.b64decode(frame_b64)
            img = Image.open(BytesIO(img_data))
            np_img = np.array(img)

            

            np_frames.append(np_img)
            timestamps.append(frame_data.timestamp)
            display_times.append(frame_data.displayTime)

        # Process batch with model (only if we have frames)
        if np_frames:
            annotated_frames, detection_texts = await detection_model.process_batch(np_frames)

            # Convert annotated frames back to base64 with timestamps and camera_id
            annotated_frames_data = []
            text_results = []

            for i, frame in enumerate(annotated_frames):
                # Convert BGR to RGB if needed
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Convert to base64
                buffer = BytesIO()
                Image.fromarray(frame).save(buffer, format="JPEG")
                img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

                # Add frame with its timestamp and camera_id
                annotated_frames_data.append({
                    "frame": f"data:image/jpeg;base64,{img_str}",
                    "timestamp": timestamps[i],
                    "displayTime": display_times[i],
                    "camera_id": camera_id
                })

                # Add text result with its timestamp and camera_id
                text_results.append({
                    "text": detection_texts[i],
                    "timestamp": timestamps[i],
                    "displayTime": display_times[i],
                    "camera_id": camera_id
                })

            processing_time = time.time() - start_time
            logger.info(f"Batch processing time: {processing_time:.3f} seconds")

            # Broadcast results via WebSocket if there are connections
            if manager.active_connections:
                # Prepare data for broadcasting
                broadcast_data = {
                    "type": "detection_results",
                    "camera_id": camera_id,
                    "timestamp": CURRENT_TIMESTAMP,
                    "user": CURRENT_USER,
                    "processing_time": f"{processing_time:.3f}s",
                    "results": [
                        {
                            "text": text_results[i]["text"],
                            "timestamp": text_results[i]["timestamp"],
                            "displayTime": text_results[i]["displayTime"]
                        } for i in range(min(5, len(text_results)))  # Send first 5 results max
                    ]
                }

                # Broadcast to all connected clients
                await manager.broadcast(broadcast_data)
                logger.info(f"Broadcast detection results to {len(manager.active_connections)} clients")

            # Return response with camera_id
            return {
                "camera_id": camera_id,
                "annotated_frames": annotated_frames_data,
                "text_results": text_results,
                "processing_time": processing_time
            }
        else:
            logger.info("No valid frames to process")
            return {
                "camera_id": camera_id,
                "annotated_frames": [],
                "text_results": [],
                "processing_time": time.time() - start_time
            }

    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

