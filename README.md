 # YOLO Inference Benchmark (Raspberry Pi 4B 8 GB)

This script benchmarks YOLOv8 models across multiple inference backends — PyTorch, ONNX Runtime, and NCNN — directly on a Raspberry Pi 4B (8 GB).
It measures both model performance (inference time) and system health metrics such as CPU usage, RAM, temperature, frequency, and throttling.

The results are automatically logged to a CSV file for easy comparison.


---

## Features

Fine-Tuned YOLOv8n (PyTorch) benchmarking

ONNX Runtime benchmarking

NCNN benchmarking (ultra-light C++ inference engine)

Logs detailed hardware metrics each run:

CPU load and frequency per core

RAM usage and availability

CPU temperature (via vcgencmd)

GPU and ARM memory split

Undervoltage / throttling flags


Average, min, max, and std-dev inference times per backend



---

## Requirements

### Python Packages

Install the required dependencies:

```bash
sudo apt update
sudo apt install python3-opencv python3-pip libraspberrypi-bin -y
pip install ultralytics onnxruntime psutil numpy ncnn
```

Tip: For best results, install onnxruntime CPU build (not GPU) — the Pi doesn’t support CUDA.




---

## System Configuration

1. Set performance mode to avoid frequency throttling:

sudo raspi-config
# → Performance Options → CPU Governor → Performance


2. Ensure sufficient GPU memory (for NCNN and OpenCV):

sudo raspi-config
# → Performance Options → GPU Memory → 128 MB


3. Monitor thermals: Use a heatsink or fan to maintain CPU temperature under 70 °C during testing.


4. Use 64-bit OS (Raspberry Pi OS 64-bit) for maximum performance and memory usage.




---

## Project Layout

model_benchmark.py       # This script
image.jpg                   # Test image
base.pt                     # YOLOv8n (PyTorch) model
pruned.pt                   # Pruned YOLOv8n model
pruned_int8.onnx            # Quantized ONNX model
pruned_int8.ncnn/           # NCNN folder with model.ncnn.param and model.ncnn.bin
benchmark_results.csv       # Output results


---

## Configuration

Edit these parameters at the top of the script to customize your benchmark:

image_path = "image.jpg"     # Path to test image
loops = 20                   # Number of inference runs per model
input_size = (640, 640)      # Model input resolution
csv_file = "benchmark_results.csv"  # Output CSV file name


---

## Run the Benchmark

Run directly from terminal:

```bash
python3 model_benchmark.py
```
###Example output:

Starting Benchmark...

Testing base.pt (PyTorch)...
Run 1/20: 305.12 ms
Run 2/20: 298.67 ms
...
base.pt (YOLOv8(PyTorch)) — Mean: 302.11 | Min: 295.47 | Max: 309.88 | Std: 4.21

When complete:

Results saved in: benchmark_results.csv


---

## Notes

Inference times on Pi 4B 8 GB typically range from:

PyTorch: 300–600 ms per frame

ONNX Runtime: 150–350 ms per frame

NCNN (INT8): 80–200 ms per frame


Performance depends on model size, quantization, and system cooling.

The CSV log can be analyzed with Excel, pandas, or Grafana to visualize performance trends.



---

## Example Use Cases

Benchmarking YOLOv8 models before deploying to IoT edge devices

Comparing PyTorch vs ONNX vs NCNN backends

Monitoring how temperature or throttling impacts inference speed
