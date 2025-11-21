import cv2
import threading
import queue
import time
import pytesseract
import csv
import os
import subprocess
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from flask import Flask, Response, render_template_string

MODEL_PATH = "pruned_int8.ncnn" 
CSV_FILE = "plate_log.csv"

CAPTURE_WIDTH = 2028
CAPTURE_HEIGHT = 1520
FRAMERATE = 30

INFERENCE_SIZE = 416 # 640, 416, 320 
CONF_THRESHOLD = 0.50  

YOLO_INTERVAL_FRAMES = 10 # Run object detection every 10 frames (3 tiems per second)

BOX_DISPLAY_TTL = 2.0 

OCR_CONFIG = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

shared_data = {
    "boxes": [],
    "text": "Scanning...", 
    "last_update": 0
}
data_lock = threading.Lock()

# Queues
inference_queue = queue.Queue(maxsize=1)
ocr_queue = queue.Queue(maxsize=1)
display_queue = queue.Queue(maxsize=2)

stop_event = threading.Event()

app = Flask(__name__)

PAGE_HTML = """
<html>
  <head>
    <title>Pi 4 LPR</title>
    <style>
      body { background-color: #222; color: #ddd; font-family: monospace; text-align: center; }
      img { border: 2px solid #555; max-width: 100%; }
      .status { margin-top: 10px; color: #0f0; }
    </style>
  </head>
  <body>
    <h1>RPi 4 - NCNN YOLO + OCR</h1>
    <img src="/video_feed">
    <div class="status">System Running...</div>
  </body>
</html>
"""

def capture_worker():
    frame_len = int(CAPTURE_WIDTH * CAPTURE_HEIGHT * 1.5)
    
    cmd = [
        "rpicam-vid", "-t", "0", "--inline",
        "--width", str(CAPTURE_WIDTH), 
        "--height", str(CAPTURE_HEIGHT),
        "--framerate", str(FRAMERATE), 
        "--codec", "yuv420", "--nopreview", "-o", "-"
    ]

    print(f"[INFO] Starting HQ Stream {CAPTURE_WIDTH}x{CAPTURE_HEIGHT}...")
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**7)
    except FileNotFoundError:
        cmd[0] = "libcamera-vid"
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**7)

    frame_counter = 0

    while not stop_event.is_set():
        try:
            raw_data = process.stdout.read(frame_len)
            if len(raw_data) != frame_len:
                break
            
            yuv_image = np.frombuffer(raw_data, dtype=np.uint8).reshape((int(CAPTURE_HEIGHT * 1.5), CAPTURE_WIDTH))
            frame = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_I420)
            
            frame_counter += 1

            if frame_counter % YOLO_INTERVAL_FRAMES == 0:
                if inference_queue.empty():
                    inference_queue.put(frame.copy())

            if not display_queue.full():
                display_queue.put(frame)

        except Exception as e:
            print(f"[ERROR] Camera: {e}")
            break
            
    process.terminate()
    print("[INFO] Capture thread stopped.")
  
def yolo_worker():
    print("[INFO] Loading NCNN Model...")
    model = YOLO(MODEL_PATH, task='detect')
    print("[INFO] YOLO Ready.")

    while not stop_event.is_set():
        try:
            frame = inference_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        results = model(frame, imgsz=INFERENCE_SIZE, conf=CONF_THRESHOLD, verbose=False)
        
        found_boxes = []
        h, w, _ = frame.shape
        
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                found_boxes.append([x1, y1, x2, y2])

                if not ocr_queue.full():
                    plate_crop = frame[y1:y2, x1:x2]
                    ocr_queue.put(plate_crop)

            with data_lock:
                shared_data['boxes'] = found_boxes
                shared_data['last_update'] = time.time()
        else:
            pass

def ocr_worker():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(["Timestamp", "Text"])

    last_logged_text = ""
    last_logged_time = 0

    print("[INFO] OCR Thread Ready.")

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
            current_time = time.time()
            
            with data_lock:
                shared_data['text'] = clean_text
                shared_data['last_update'] = current_time 

            if (clean_text != last_logged_text) or (current_time - last_logged_time > 5.0):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(CSV_FILE, 'a', newline='') as f:
                    csv.writer(f).writerow([timestamp, clean_text])
                print(f"[LOG] {clean_text}")
                last_logged_text = clean_text
                last_logged_time = current_time

def generate_frames():
    box_color = (0, 255, 0)
    
    while True:
        try:
            frame = display_queue.get(timeout=1.0)
            
            display_h, display_w = frame.shape[:2]
            target_w = 1024
            target_h = int(target_w * (display_h / display_w))
            
            view_frame = cv2.resize(frame, (target_w, target_h))
            
            scale_x = target_w / display_w
            scale_y = target_h / display_h

            with data_lock:
                time_since_detection = time.time() - shared_data['last_update']
                
                if time_since_detection < BOX_DISPLAY_TTL:
                    boxes = shared_data['boxes']
                    text = shared_data['text']
                    
                    for (x1, y1, x2, y2) in boxes:
                        sx1, sy1 = int(x1 * scale_x), int(y1 * scale_y)
                        sx2, sy2 = int(x2 * scale_x), int(y2 * scale_y)

                        cv2.rectangle(view_frame, (sx1, sy1), (sx2, sy2), box_color, 2)
                        
                        label = f"{text}"
                        cv2.rectangle(view_frame, (sx1, sy1 - 25), (sx1 + 200, sy1), box_color, -1)
                        cv2.putText(view_frame, label, (sx1, sy1 - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                else:
                    if time_since_detection > BOX_DISPLAY_TTL + 0.5:
                        shared_data['text'] = "Scanning..."

            ret, buffer = cv2.imencode('.jpg', view_frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except queue.Empty: continue
        except Exception: break

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    t1 = threading.Thread(target=capture_worker, daemon=True)
    t2 = threading.Thread(target=yolo_worker, daemon=True)
    t3 = threading.Thread(target=ocr_worker, daemon=True)

    t1.start()
    time.sleep(1)
    t2.start()
    t3.start()

    print(f"[INFO] Web Server running at http://0.0.0.0:5000")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        print("[INFO] Shutting down.")
