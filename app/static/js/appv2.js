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
    let displayQueue = []; // Priority queue for batch results
    let isProcessingStopped = false; // Used to halt rendering after stopping
    const BATCH_TIMEOUT = 60000; // 60 seconds timeout
    const STALE_THRESHOLD = 5000;

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
            alert("Please enter a username");
            return;
        }
        isProcessingStopped = false; // Reset stop flag

        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'environment'
                }
            });

            videoElement.srcObject = stream;

            videoElement.onloadedmetadata = () => {
                isCapturing = true;
                captureInterval = setInterval(captureFrame, 1000 / targetFPS);
                fpsUpdateInterval = setInterval(updateFPS, 1000);
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
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }

        isCapturing = false;
        isProcessingStopped = true; // Set stop flag to prevent further processing
        clearInterval(captureInterval);
        clearInterval(fpsUpdateInterval);

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
        captureCanvas.toBlob(blob => {
            frameBuffer.push({ blob, timestamp });
            queueSizeElement.textContent = frameBuffer.length;
        
            if (frameBuffer.length >= batchSize) {
                processFrameBatch(); // Call batch processing here
            }
        }, 'image/jpeg', 0.8);
        
        queueSizeElement.textContent = frameBuffer.length;

        if (frameBuffer.length >= batchSize) {
            processFrameBatch();
        }
    }

    async function processFrameBatch() {
    const frameBatch = frameBuffer.splice(0, batchSize);
    const batchId = Date.now();
    const username = usernameInput.value.trim();
    const cameraNumber = cameraNumberSelect.value;
    const now = Date.now();

    const formData = new FormData();
    formData.append('username', username);
    formData.append('camera_number', cameraNumber);
    formData.append('batch_id', batchId);

    frameBatch.forEach(({ blob, timestamp }, index) => {
        formData.append(`frame_${index}`, blob, `frame_${index}.jpg`);
        formData.append(`timestamp_${index}`, timestamp);
    });

    try {
        statusText.textContent = 'Processing...';

        const response = await fetch('/api/v2/process_batch', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        batchQueue.push({ id: data.batch_id, timestamp: now });
        batchQueue.sort((a, b) => a.id - b.id);
        requestBatchResult(batchQueue[0], username, cameraNumber);
    } catch (error) {
        console.error('Error processing frames:', error);
        statusText.textContent = 'Processing Error';
        statusText.className = 'error';
    }
    }

    async function requestBatchResult(batchEntry, username, cameraNumber) {
        const { id: batchId, timestamp } = batchEntry;
    
        if (batchIdsWaiting.has(batchId)) return;
    
        const now = Date.now();
        const age = now - timestamp;
    
        if (age > STALE_THRESHOLD) {
            console.warn(`Skipping stale batch ${batchId}, in queue for ${age}ms`);
            batchQueue.shift();
            fetchNextBatch();
            return;
        }
    
        batchIdsWaiting.add(batchId);
    
        const timeout = setTimeout(() => {
            batchIdsWaiting.delete(batchId);
            console.warn(`Timeout reached for batch ${batchId}`);
            statusText.textContent = 'Timeout Error';
            statusText.className = 'error';
        }, BATCH_TIMEOUT);
    
        try {
            const formData = new FormData();
            formData.append('username', username);
            formData.append('camera_number', cameraNumber);
            formData.append('batch_id', batchId);
    
            const response = await fetch('/api/v2/get_result', {
                method: 'POST',
                body: formData
            });
    
            clearTimeout(timeout);
    
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
    
            const data = await response.json();
    
            if (data.results) {
                data.results.forEach(result => {
                    displayQueue.push({
                        imgSrc: `data:image/jpeg;base64,${result.annotated_image_b64}`,
                        frameNumber: result.frame_number,
                        timestamp: result.timestamp,
                        caption: result.caption
                    });
                });
    
                batchIdsWaiting.delete(batchId);
                batchQueue.shift();
                fetchNextBatch();
            }
    
            statusText.textContent = 'Active';
            statusText.className = 'active';
        } catch (error) {
            console.error('Error retrieving batch results:', error);
            statusText.textContent = 'Retrieval Error';
            statusText.className = 'error';
            batchIdsWaiting.delete(batchId);
        }
    }
    
    
    function renderFramesLoop() {
        if(displayQueue.length > 0 && !isProcessingStopped) {
            const { imgSrc, frameNumber, timestamp, caption } = displayQueue.shift();

            const img = new Image();
            img.onload = () => {
                previewContext.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
                previewContext.drawImage(img, 0, 0, previewCanvas.width, previewCanvas.height);
    
                previewContext.fillStyle = 'rgba(0, 0, 0, 0.5)';
                previewContext.fillRect(10, previewCanvas.height - 70, 320, 50);
                previewContext.fillStyle = 'white';
                previewContext.font = '14px Arial';
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
                console.error("Error loading image.");
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