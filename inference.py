import cv2
import threading
import queue
import time
import pytesseract
import csv
import os
from datetime import datetime
from ultralytics import YOLO
from flask import Flask, Response, render_template_string

# ================= CONFIGURATION =================
MODEL_PATH = "yolov8n_ncnn_int8" 
CSV_FILE = "plate_log.csv"

# Camera Settings
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
INFERENCE_SIZE = 640
CONF_THRESHOLD = 0.50  

OCR_CONFIG = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# ================= FLASK SETUP =================
app = Flask(__name__)

# HTML Template for the webpage
PAGE_HTML = """
<html>
  <head>
    <title>Pi 4 LPR Stream</title>
    <style>
      body { background-color: #333; color: white; font-family: sans-serif; text-align: center; }
      h1 { margin-top: 20px; }
      img { border: 5px solid #444; max-width: 100%; }
      .data { margin-top: 20px; font-size: 20px; color: #0f0; }
    </style>
  </head>
  <body>
    <h1>Raspberry Pi LPR - YOLO INT8</h1>
    <img src="/video_feed">
    <div class="data">Latest Plate: <span id="plate">Waiting...</span></div>
  </body>
</html>
"""

# ================= SHARED RESOURCES =================
frame_queue = queue.Queue(maxsize=1)
ocr_queue = queue.Queue(maxsize=1)
display_queue = queue.Queue(maxsize=1)

stop_event = threading.Event()

latest_plate_text = "Scanning..."
text_lock = threading.Lock()

last_logged_text = ""
last_logged_time = 0

# ================= CSV INIT =================
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Detected Text"])

# ================= THREAD 1: CAPTURE =================
def capture_worker():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame)
    cap.release()

# ================= THREAD 2: YOLO + DRAWING =================
def yolo_worker():
    model = YOLO(MODEL_PATH, task='detect')
    box_color = (0, 255, 0)

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        results = model(frame, imgsz=INFERENCE_SIZE, conf=CONF_THRESHOLD, verbose=False)
        boxes = results[0].boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if not ocr_queue.full():
                plate_crop = frame[y1:y2, x1:x2]
                ocr_queue.put(plate_crop)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
            
            with text_lock:
                current_text = latest_plate_text
            
            label = f"{current_text}"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, (x1, y1 - t_size[1] - 10), (x1 + t_size[0], y1), box_color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        if not display_queue.empty():
            try:
                display_queue.get_nowait()
            except queue.Empty:
                pass
        display_queue.put(frame)

# ================= THREAD 3: OCR + LOGGING =================
def ocr_worker():
    global latest_plate_text, last_logged_text, last_logged_time
    while not stop_event.is_set():
        try:
            plate_img = ocr_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        text = pytesseract.image_to_string(thresh, config=OCR_CONFIG)
        clean_text = "".join(text.split()).strip()
        
        if len(clean_text) >= 3:
            with text_lock:
                latest_plate_text = clean_text

            current_time = time.time()
            if (clean_text != last_logged_text) or (current_time - last_logged_time > 5.0):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(CSV_FILE, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, clean_text])
                print(f"[LOGGED] {timestamp} - {clean_text}")
                last_logged_text = clean_text
                last_logged_time = current_time

# ================= WEB SERVER LOGIC =================
@app.route('/')
def index():
    return render_template_string(PAGE_HTML)

def generate_frames():
    while True:
        try:
            # Get processed frame from YOLO thread
            frame = display_queue.get(timeout=1.0)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Create MJPEG stream format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Stream Error: {e}")
            break

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ================= MAIN =================
if __name__ == "__main__":
    # Start background threads
    t1 = threading.Thread(target=capture_worker, daemon=True)
    t2 = threading.Thread(target=yolo_worker, daemon=True)
    t3 = threading.Thread(target=ocr_worker, daemon=True)

    t1.start()
    time.sleep(1.0)
    t2.start()
    t3.start()

    print(f"[INFO] Starting Web Server on 0.0.0.0:5000")
    print(f"[INFO] View the feed at http://<YOUR_PI_IP>:5000")
    
    # Run Flask (this blocks the main thread, which is what we want)
    # host='0.0.0.0' makes it accessible from other computers
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        # No need to join threads here as they are daemons
        print("[INFO] Server Stopped.")
