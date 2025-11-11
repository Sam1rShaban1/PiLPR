import time
import numpy as np
import cv2
import csv
import psutil
import subprocess
from ultralytics import YOLO
import onnxruntime as ort
import ncnn

# -------------------------------------
# Config

image_path = "image.jpg"
loops = 20  # number of inference runs to average
input_size = (640, 640)
csv_file = "benchmark_results.csv"

image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"❌ Could not load image at {image_path}")


def preprocess(img):
    img_resized = cv2.resize(img, input_size)
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_resized = img_resized.astype(np.float32) / 255.0
    img_resized = np.transpose(img_resized, (2, 0, 1))  # CHW
    img_resized = np.expand_dims(img_resized, 0)  # NCHW
    return img_resized

# -------------------------------------
# Metrics

def parse_throttled_output(val):
    flags = int(val.strip().split("=")[-1], 16)
    return {
        "under_voltage": bool(flags & 0x1),
        "arm_freq_capped": bool(flags & 0x2),
        "currently_throttled": bool(flags & 0x4),
        "soft_temp_limit": bool(flags & 0x8),
        "under_voltage_occurred": bool(flags & 0x10000),
        "freq_capped_occurred": bool(flags & 0x20000),
        "throttled_occurred": bool(flags & 0x40000),
        "soft_temp_limit_occurred": bool(flags & 0x80000),
    }


def get_system_metrics():
    metrics = {}

    # CPU usage
    metrics["cpu_usage_percent"] = psutil.cpu_percent(interval=None)

    # Per-core frequency
    freqs = psutil.cpu_freq(percpu=True)
    for i, f in enumerate(freqs):
        metrics[f"cpu{i}_freq_mhz"] = f.current

    # RAM
    vm = psutil.virtual_memory()
    metrics["ram_used_mb"] = vm.used / (1024 * 1024)
    metrics["ram_available_mb"] = vm.available / (1024 * 1024)
    metrics["ram_percent"] = vm.percent

    # Temp
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
        metrics["cpu_temp_c"] = float(out.replace("temp=", "").replace("'C\n", ""))
    except Exception:
        metrics["cpu_temp_c"] = np.nan

    # Core freq
    try:
        arm = subprocess.check_output(["vcgencmd", "measure_clock", "arm"]).decode()
        core = subprocess.check_output(["vcgencmd", "measure_clock", "core"]).decode()
        metrics["arm_freq_mhz"] = int(arm.split("=")[1]) / 1_000_000
        metrics["core_freq_mhz"] = int(core.split("=")[1]) / 1_000_000
    except Exception:
        metrics["arm_freq_mhz"] = np.nan
        metrics["core_freq_mhz"] = np.nan

    # GPU memory info
    try:
        gpu_mem = subprocess.check_output(["vcgencmd", "get_mem", "gpu"]).decode()
        arm_mem = subprocess.check_output(["vcgencmd", "get_mem", "arm"]).decode()
        metrics["gpu_mem_mb"] = int(gpu_mem.split("=")[1].replace("M\n", ""))
        metrics["arm_mem_mb"] = int(arm_mem.split("=")[1].replace("M\n", ""))
    except Exception:
        metrics["gpu_mem_mb"] = np.nan
        metrics["arm_mem_mb"] = np.nan

    # Throttling info
    try:
        throttled = subprocess.check_output(["vcgencmd", "get_throttled"]).decode()
        throttle_flags = parse_throttled_output(throttled)
        metrics.update(throttle_flags)
    except Exception:
        for k in [
            "under_voltage",
            "arm_freq_capped",
            "currently_throttled",
            "soft_temp_limit",
            "under_voltage_occurred",
            "freq_capped_occurred",
            "throttled_occurred",
            "soft_temp_limit_occurred",
        ]:
            metrics[k] = np.nan

    return metrics


def save_run_to_csv(writer, model_name, backend, run_idx, t_infer, metrics):
    row = {
        "Model": model_name,
        "Backend": backend,
        "Run": run_idx,
        "Time (ms)": f"{t_infer:.2f}",
        **metrics,
    }
    writer.writerow(row)

def summarize(times, model_name, backend, writer):
    mean_t, min_t, max_t, std_t = np.mean(times), np.min(times), np.max(times), np.std(times)
    writer.writerow(
        {
            "Model": model_name,
            "Backend": backend,
            "Run": "Average",
            "Time (ms)": f"{mean_t:.2f}",
        }
    )
    print(f"✅ {model_name} ({backend}) — Mean: {mean_t:.2f} | Min: {min_t:.2f} | Max: {max_t:.2f} | Std: {std_t:.2f}")

# -------------------------------------
# Inference

def test_yolo_pt(model_path, writer):
    model_name = model_path.split("/")[-1]
    model = YOLO(model_path)
    print(f"\n🚀 Testing {model_name} (PyTorch)...")
    model.predict(source=image_path, imgsz=640)  # warmup

    times = []
    for i in range(loops):
        metrics_before = get_system_metrics()
        t1 = time.time()
        model.predict(source=image_path, imgsz=640, verbose=False)
        t_infer = (time.time() - t1) * 1000
        metrics_after = get_system_metrics()

        for k in metrics_before:
            metrics_before[k] = (metrics_before[k] + metrics_after[k]) / 2

        times.append(t_infer)
        save_run_to_csv(writer, model_name, "YOLOv8(PyTorch)", i + 1, t_infer, metrics_before)
        print(f"Run {i+1}/{loops}: {t_infer:.2f} ms")

    summarize(times, model_name, "YOLOv8(PyTorch)", writer)


def test_onnx(model_path, writer):
    model_name = model_path.split("/")[-1]
    print(f"\n🚀 Testing {model_name} (ONNX)...")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    img = preprocess(image)
    session.run(None, {input_name: img})  # warmup

    times = []
    for i in range(loops):
        metrics_before = get_system_metrics()
        t1 = time.time()
        session.run(None, {input_name: img})
        t_infer = (time.time() - t1) * 1000
        metrics_after = get_system_metrics()

        for k in metrics_before:
            metrics_before[k] = (metrics_before[k] + metrics_after[k]) / 2

        times.append(t_infer)
        save_run_to_csv(writer, model_name, "ONNXRuntime", i + 1, t_infer, metrics_before)
        print(f"Run {i+1}/{loops}: {t_infer:.2f} ms")

    summarize(times, model_name, "ONNXRuntime", writer)


def test_ncnn(model_dir, writer):
    model_name = model_dir.rstrip("/").split("/")[-1]
    print(f"\n🚀 Testing {model_name} (NCNN)...")
    net = ncnn.Net()
    net.load_param(f"{model_dir}/model.param")
    net.load_model(f"{model_dir}/model.bin")

    mat_in = ncnn.Mat.from_pixels_resize(
        image,
        ncnn.Mat.PixelType.PIXEL_BGR2RGB,
        image.shape[1],
        image.shape[0],
        640,
        640,
    )

    # warmup
    extractor = net.create_extractor()
    extractor.input("images", mat_in)
    _, _ = extractor.extract("output0")

    times = []
    for i in range(loops):
        metrics_before = get_system_metrics()
        t1 = time.time()
        extractor = net.create_extractor()
        extractor.input("images", mat_in)
        _, _ = extractor.extract("output0")
        t_infer = (time.time() - t1) * 1000
        metrics_after = get_system_metrics()

        for k in metrics_before:
            metrics_before[k] = (metrics_before[k] + metrics_after[k]) / 2

        times.append(t_infer)
        save_run_to_csv(writer, model_name, "NCNN", i + 1, t_infer, metrics_before)
        print(f"Run {i+1}/{loops}: {t_infer:.2f} ms")

    summarize(times, model_name, "NCNN", writer)



if __name__ == "__main__":
    print("Starting Benchmark...\n")
    metric_keys = list(get_system_metrics().keys())
    fieldnames = ["Model", "Backend", "Run", "Time (ms)"] + metric_keys

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        test_yolo_pt("yolov8n.pt", writer)
        test_yolo_pt("yolov8n_pruned.pt", writer)
        test_onnx("model.onnx", writer)
        try:
            test_ncnn("model.ncnn", writer)
        except Exception as e:
            print(f"\nNCNN test skipped or failed: {e}")

    print(f"\n Results saved in: {csv_file}")
