document.addEventListener('DOMContentLoaded', () => {
    // Elements and state setup
    const videoElement = document.getElementById('camera-feed');
    const previewCanvas = document.getElementById('preview-canvas');
    const previewContext = previewCanvas.getContext('2d');
    const startButton = document.getElementById('start-btn');
    const stopButton = document.getElementById('stop-btn');
    const fpsSelect = document.getElementById('fps-setting');
    const maxResultsSelect = document.getElementById('max-results');
    const statusText = document.getElementById('status-text');
    const fpsCounter = document.getElementById('fps-counter');
    const detectionResults = document.getElementById('detection-results');
    const processingTimeElement = document.getElementById('processing-time');
    const queueSizeElement = document.getElementById('queue-size');
    const delayElement = document.getElementById('delay-time'); // New element for delay display
    const usernameInput = document.getElementById('username');
    const cameraNumberSelect = document.getElementById('camera-number');
    const captureCanvas = document.createElement('canvas');
    const captureContext = captureCanvas.getContext('2d');
    captureCanvas.width = 640;
    captureCanvas.height = 360;

    // App state
    let stream = null;
    let isCapturing = false;
    let captureInterval = null;
    let frameBuffer = [];
    let batchQueue = [];
    let frameCount = 0;
    let fpsUpdateInterval = null;
    let fetchResultInterval = null; // New interval for fetching results
    let displayQueue = []; // Priority queue for batch results
    let isProcessingStopped = false; // Used to halt rendering after stopping
    const BATCH_TIMEOUT = 60000; // 60 seconds timeout
    const STALE_THRESHOLD = 50000; // 5 seconds

    // Config
    let targetFPS = parseInt(fpsSelect.value);
    let batchSize = targetFPS;
    let maxResults = parseInt(maxResultsSelect.value || '10');
    let renderInterval = 1000 / targetFPS;
    let batchIdsWaiting = new Set();

    // Initialize canvas sizes
    previewCanvas.width = 640;
    previewCanvas.height = 360;

    // Event Listeners
    startButton.addEventListener('click', startCapture);
    stopButton.addEventListener('click', stopCapture);

    fpsSelect.addEventListener('change', () => {
        targetFPS = parseInt(fpsSelect.value);
        batchSize = targetFPS;
        renderInterval = 1000 / targetFPS;
        if (isCapturing) {
            clearInterval(captureInterval);
            captureInterval = setInterval(captureFrame, 1000 / targetFPS);
        }
    });

    maxResultsSelect.addEventListener('change', () => {
        maxResults = parseInt(maxResultsSelect.value);
        updateResultsList(true);
    });

    async function startCapture() {
        const username = usernameInput.value.trim();
        const cameraNumber = cameraNumberSelect.value;
        if (!username) {
            alert('Please enter a username');
            return;
        }
        isProcessingStopped = false; // Reset stop flag

        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'environment',
                },
            });

            videoElement.srcObject = stream;

            videoElement.onloadedmetadata = () => {
                isCapturing = true;
                captureInterval = setInterval(captureFrame, 1000 / targetFPS);
                fpsUpdateInterval = setInterval(updateFPS, 1000);
                fetchResultInterval = setInterval(fetchNextBatch, 1000); // Fetch results every 1 second
                renderFramesLoop();

                startButton.disabled = true;
                stopButton.disabled = false;
                statusText.textContent = 'Active';
                statusText.className = 'active';
            };
        } catch (error) {
            console.error('Error accessing camera:', error);
            statusText.textContent = 'Camera Error';
            statusText.className = 'error';
        }
    }

    function stopCapture() {
        if (stream) {
            stream.getTracks().forEach((track) => track.stop());
            stream = null;
        }

        isCapturing = false;
        isProcessingStopped = true; // Set stop flag to prevent further processing
        clearInterval(captureInterval);
        clearInterval(fpsUpdateInterval);
        clearInterval(fetchResultInterval); // Clear the fetch result interval

        startButton.disabled = false;
        stopButton.disabled = true;
        statusText.textContent = 'Stopped';
        statusText.className = '';

        frameBuffer = [];
        displayQueue = [];
        batchQueue = [];

        previewContext.fillStyle = 'black';
        previewContext.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
        previewContext.fillStyle = 'white';
        previewContext.font = '20px Arial';
        previewContext.textAlign = 'center';
        previewContext.fillText('Camera Stopped', previewCanvas.width / 2, previewCanvas.height / 2);
    }

    function captureFrame() {
        if (!isCapturing || !videoElement.videoWidth) return;
        captureContext.drawImage(videoElement, 0, 0, captureCanvas.width, captureCanvas.height);

        const timestamp = Date.now();
        const frameData = captureCanvas.toDataURL('image/jpeg', 0.8);

        frameBuffer.push({
            frame: frameData,
            timestamp: timestamp,
        });
        queueSizeElement.textContent = frameBuffer.length;

        if (frameBuffer.length >= batchSize) {
            processFrameBatch();
        }
    }

    async function processFrameBatch() {
        const frameBatch = frameBuffer.splice(0, batchSize);
        const batchId = Date.now(); // Use timestamp as batch ID
        const username = usernameInput.value.trim();
        const cameraNumber = cameraNumberSelect.value;
        const now = Date.now();

        try {
            statusText.textContent = 'Processing...';

            const response = await fetch('/api/process_batch', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    username: username,
                    camera_number: cameraNumber,
                    batch_id: batchId,
                    frames: frameBatch.map((frame, index) => ({
                        frame_number: index,
                        image_b64: frame.frame.replace(/^data:image\/jpeg;base64,/, ''),
                        timestamp: frame.timestamp,
                    })),
                }),
            });

            const data = await response.json();

            batchQueue.push({ id: data.batch_id, timestamp: now });
            batchQueue.sort((a, b) => a.id - b.id); // Sort based on batch ID

        } catch (error) {
            console.error('Error processing frames:', error);
            statusText.textContent = 'Processing Error';
            statusText.className = 'error';
        }
    }

    async function requestBatchResult(batchEntry, username, cameraNumber) {
        const { id: batchId, timestamp } = batchEntry;

        // if (batchIdsWaiting.has(batchId)) return; // Prevent duplicate requests

        const now = Date.now();
        const age = now - timestamp;

        if (age > STALE_THRESHOLD) {
            console.warn(`Skipping stale batch ${batchId}, in queue for ${age}ms`);
            batchQueue.shift(); // Remove it from the queue
            return; // Just return since fetchNextBatch will be called again in next interval
        }

        batchIdsWaiting.add(batchId);

        const timeout = setTimeout(() => {
            batchIdsWaiting.delete(batchId);
            console.warn(`Timeout reached for batch ${batchId}`);
            statusText.textContent = 'Timeout Error';
            statusText.className = 'error';
        }, BATCH_TIMEOUT);

        try {
            const response = await fetch('/api/get_result', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    username: username,
                    camera_number: cameraNumber,
                    batch_id: batchId,
                }),
            });

            clearTimeout(timeout);

            if (!response.ok) {
                if (response.status === 404) {
                    // Handle 404 error specifically
                    console.error(`Error: Resource not found for batch ${batchId}`);
                    statusText.textContent = `Error: Resource not found (404) for batch ${batchId}`;
                    statusText.className = 'error';
                    
                } else {
                    // Handle other errors (e.g., 500, 400, etc.)
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return; // Exit the function early if there's an error
            }

            const data = await response.json();

            if (data.results) {
                data.results.forEach((result) => {
                    displayQueue.push({
                        imgSrc: `data:image/jpeg;base64,${result.annotated_image_b64}`,
                        frameNumber: result.frame_number,
                        timestamp: result.timestamp,
                        caption: result.caption,
                    });
                });

                batchIdsWaiting.delete(batchId);
                batchQueue.shift(); 
            }

            statusText.textContent = 'Active';
            statusText.className = 'active';
        } catch (error) {
            console.error('Error retrieving batch results:', error);
            statusText.textContent = 'Retrieval Error';
            statusText.className = 'error';
            batchIdsWaiting.delete(batchId);
        }

        const delay = (Date.now() - timestamp) / 1000; // Calculate delay in seconds
        delayElement.textContent = `Delay: ${delay.toFixed(1)}s`; // Display delay
    }

    function renderFramesLoop() {
        if (displayQueue.length > 0 && !isProcessingStopped) {
            const { imgSrc, frameNumber, timestamp, caption } = displayQueue.shift();

            const img = new Image();
            img.onload = () => {
                previewContext.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
                previewContext.drawImage(img, 0, 0, previewCanvas.width, previewCanvas.height);

                previewContext.fillStyle = 'rgba(0, 0, 0, 0.5)';
                previewContext.fillRect(10, previewCanvas.height - 70, 320, 70);
                previewContext.fillStyle = 'white';
                previewContext.font = '14px Arial';
                previewContext.textAlign = 'left';
                previewContext.fillText(`Frame: ${frameNumber}`, 15, previewCanvas.height - 55);
                previewContext.fillText(`Time: ${new Date(timestamp).toLocaleTimeString()}`, 15, previewCanvas.height - 35);
                previewContext.fillText(`Caption: ${caption}`, 15, previewCanvas.height - 15);

                const resultItem = document.createElement('div');
                resultItem.className = 'result-item';
                resultItem.innerHTML = `<strong>Frame ${frameNumber}</strong>: Timestamp: ${new Date(timestamp).toLocaleTimeString()}<br>Caption: ${caption}`;
                detectionResults.insertBefore(resultItem, detectionResults.firstChild);

                // Keep only the latest maxResults captions
                while (detectionResults.childNodes.length > maxResults) {
                    detectionResults.removeChild(detectionResults.lastChild);
                }
            };
            img.onerror = () => {
                console.error('Error loading image.');
            };
            img.src = imgSrc;
        }

        if (!isProcessingStopped) {
            setTimeout(renderFramesLoop, renderInterval);
        }
    }

    function fetchNextBatch() {
        if (batchQueue.length > 0) {
            const nextBatchId = batchQueue[0];
            const username = usernameInput.value.trim();
            const cameraNumber = cameraNumberSelect.value;
            requestBatchResult(nextBatchId, username, cameraNumber);
        }
    }

    function updateFPS() {
        const fps = frameCount;
        fpsCounter.textContent = `${fps} FPS`;
        frameCount = 0;
    }
});