# Raspberry Pi 4 License Plate Recognition (LPR)

This project implements a real-time License Plate Recognition system optimized for the Raspberry Pi 4 Model B. It utilizes a custom **YOLOv8 model quantized to Int8** running on the **NCNN framework** for object detection, combined with Tesseract OCR for text extraction.

The system employs a non-blocking, multi-threaded architecture to maintain a smooth video feed (30 FPS) while performing heavy AI inference asynchronously.

## System Architecture

The application uses a **Producer-Consumer** pattern distributed across three daemon threads to maximize CPU core utilization on the Raspberry Pi 4.

1.  **Capture Thread (Producer):**
    *   Spawns a native subprocess (`rpicam-vid`) to access the HQ Camera directly, bypassing standard OpenCV driver overhead.
    *   Reads raw **YUV420** bytes from `stdout`.
    *   Performs manual YUV-to-BGR conversion using NumPy (highly optimized C-bindings).
    *   Pushes frames to the display queue (30 FPS) and inference queue (regulated).

2.  **Inference Thread (Consumer 1):**
    *   Polls the inference queue.
    *   **Skip Logic:** Processes only every *N*th frame (configurable) to prevent CPU saturation.
    *   **Backend:** Uses Ultralytics YOLOv8 with NCNN (Vulkan/CPU optimized).
    *   **Resizing:** Downscales input to 320x320 for detection speed, but maps coordinates back to the source resolution (2028x1520) for high-quality cropping.

3.  **OCR Thread (Consumer 2):**
    *   triggered only when a bounding box is detected.
    *   Receives the full-resolution crop of the license plate.
    *   Applies OpenCV pre-processing (Grayscale -> Upscale -> Otsu Thresholding).
    *   Runs Tesseract LSTM engine.
    *   Logs unique strings to CSV with timestamp logic (prevents duplicate logging within a time window).

4.  **Display Loop (Main Thread):**
    *   Runs a Flask web server.
    *   Retrieves the latest frame from the capture thread.
    *   **TTL Rendering:** Draws bounding boxes and text only if the detection timestamp is within the valid Time-To-Live (TTL) window.
    *   Resizes the output stream to 1024px width to reduce network bandwidth usage while maintaining internal high-resolution processing.

## Hardware Specifications

*   **Device:** Raspberry Pi 4 Model B (8 GB RAM recommended).
*   **Camera:** Raspberry Pi HQ Camera (Sony IMX477).
*   **Storage:** Class 10 / UHS-I MicroSD (High random I/O performance recommended).
*   **Cooling:** Active cooling (Fan) is mandatory. NCNN inference utilizes NEON vector instructions which generate significant heat (approx. 75°C peak).

## Requirements

### System Dependencies
```bash
sudo apt update && sudo apt upgrade -y
# Camera and Graphics libraries
sudo apt install libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev -y
# OCR Engine
sudo apt install tesseract-ocr libtesseract-dev -y
```

### Python Dependencies
It is recommended to run this project in a virtual environment.
```bash
pip install ultralytics opencv-python-headless pytesseract flask psutil numpy ncnn
```

## Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Sam1rShaban1/PiLPR.git
    cd PiLPR
    ```

2.  **Model Setup:**
    Ensure your NCNN model folder (containing `model.ncnn.bin`, `model.ncnn.param``) is placed in the root directory. Update `MODEL_PATH` in `inference.py` if named differently.

3.  **Run the Inference System:**
    ```bash
    python3 inference.py
    ```

4.  **Access the Stream:**
    The video feed is served via Flask. Access it at: `http://<RASPBERRY_PI_IP>:5000`

## Developing Custom Models

To replicate the performance achieved in this project, you must export your YOLOv8 model specifically for NCNN with Int8 quantization.

1.  **Train your model** (standard YOLOv8 PyTorch training).
2.  **Export to NCNN:**
    ```bash
    yolo export model=best.pt format=ncnn half=True
    ```
    *Note: For Int8 quantization, you typically need to perform Post-Training Quantization (PTQ) using a calibration dataset. Refer to the Ultralytics documentation for PQT.*

## Benchmark Results

The following statistics represent an average across three Raspberry Pi 4 Model B (8GB) devices running at 1.80 GHz with an input resolution of 640x640.

### Detailed Performance Statistics

| Model Format | Backend | Avg Time | Min Time | Max Time | Std Dev | FPS (Approx) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **pruned_int8.ncnn** | **NCNN** | **379.9 ms** | **373.4 ms** | **401.5 ms** | **5.12 ms** | **2.63** |
| pruned_int8.onnx | ONNX Runtime | 487.9 ms | 480.1 ms | 615.0 ms | 6.25 ms | 2.05 |
| pruned.pt | PyTorch | 988.1 ms | 980.3 ms | 1002.9 ms | 4.21 ms | 1.01 |
| base.pt | PyTorch | 988.6 ms | 974.9 ms | 1012.7 ms | 4.55 ms | 1.01 |

### Backend Analysis

*   **NCNN (Int8):** Averaged **55% CPU usage**. This low overhead is critical for allowing the Camera and OCR threads to run without frame drops. Peak thermal load was 75°C.
*   **ONNX Runtime:** Averaged **95% CPU usage**. While faster than PyTorch, it saturates the CPU cores, causing potential jitter in video capture threads.
*   **PyTorch:** Averaged 35% CPU usage but suffered from massive framework overhead, resulting in unusable framerates (~1 FPS).

## Troubleshooting

*   **"Bad new number of rows in function 'reshape'":**
    This occurs if using `cv2.VideoCapture` with `libcamerify` on Bullseye/Bookworm OS due to stride alignment issues. This project uses `subprocess` with `rpicam-vid` to bypass this error completely.

*   **Camera not detected:**
    Ensure the ribbon cable is seated correctly (blue side facing the USB ports on Pi 4, silver contacts facing the HDMI ports). Run `libcamera-hello --list-cameras` to verify.

*   **Low FPS on Web Stream:**
    The inference script runs at 30 FPS internally. If the web stream lags, it is likely network bandwidth. The script includes a resize step (`target_w=1024`) in `generate_frames` to mitigate this.

*   **Thermal Throttling:**
    If the Pi slows down after 5-10 minutes, check `vcgencmd measure_temp`. If it exceeds 80°C, the CPU is throttling. Install a heatsink or fan.

## File Structure

```text
.
├── inference.py            # Main multi-threaded application
├── model_benchmark.py      # Hardware performance testing tool
├── plate_log.csv           # Output log
├── pruned_int8.ncnn/       # NCNN Model assets
│   ├── model.ncnn.param
│   ├── model.ncnn.bin
│   └── metadata.yaml
└── README.md
```
