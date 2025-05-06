
# Video upload endpoint
@app.post("/api/upload-video")
async def upload_video(
        video: UploadFile = File(...),
        cameraId: str = Form("camera-1"),
        user: str = Form(CURRENT_USER),
        frame_rate: int = Form(5)
):
    """Handle video uploads for AI processing"""
    try:
        logger.info(f"Received video upload: {video.filename} from {user} for camera {cameraId}")

        # Create temp directory to store the uploaded file
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded file to temp location
            temp_path = os.path.join(temp_dir, secure_filename(video.filename))
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(video.file, buffer)

            # Process the video
            start_time = time.time()
            detections, video_length = await process_video_with_ai(temp_path, frame_rate)
            processing_time = time.time() - start_time

            # Generate metadata with sequence number
            sequence_num = await get_next_sequence()

            # Create metadata entry
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "cameraId": cameraId,
                "user": user,
                "sequence": sequence_num,
                "detections": detections,
                "processingStats": {
                    "processingTime": processing_time,
                    "videoLength": video_length,
                    "frameRate": frame_rate
                }
            }

            # Save metadata
            await save_metadata(metadata)

            return JSONResponse({
                "success": True,
                "detections": len(detections),
                "processingTime": processing_time * 1000,  # Convert to ms
                "videoLength": video_length,
                "sequence": sequence_num
            })

    except Exception as e:
        logger.error(f"Error processing uploaded video: {str(e)}")
        return JSONResponse({
            "success": False,
            "message": f"Error processing video: {str(e)}"
        }, status_code=500)


async def process_video_with_ai(video_path: str, frame_rate: int) -> tuple:
    """Process video file and return detections and video length."""
    try:
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Unable to open video file")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_length = frame_count / fps if fps > 0 else 0

        detections = []
        frame_batch = []
        frame_counter = 0
        batch_size = BATCH_SIZE
        frame_interval = int(fps / frame_rate) if fps > 0 else 1

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process only every frame_interval frame
            if frame_counter % frame_interval == 0:
                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_batch.append(frame_rgb)

            frame_counter += 1

            # Process batch when full or at end
            if len(frame_batch) >= batch_size or not ret:
                if frame_batch:
                    # Convert to PIL Images for YOLO
                    pil_images = [Image.fromarray(f) for f in frame_batch]
                    results = model.predict(pil_images)

                    # Extract detections
                    for i, result in enumerate(results):
                        frame_detections = []
                        for box in result.boxes:
                            label = model.names[int(box.cls)]
                            confidence = float(box.conf)
                            frame_detections.append({
                                "label": label,
                                "confidence": confidence,
                                "timestamp": (frame_counter - len(frame_batch) + i) / fps if fps > 0 else frame_counter
                            })
                        detections.extend(frame_detections)

                    frame_batch = []

        cap.release()
        return detections, video_length

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise

