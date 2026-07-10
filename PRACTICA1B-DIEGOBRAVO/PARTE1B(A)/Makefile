BIN     := ./build/yolo_benchmark
VIDEO   := videos/videoPrueba.mp4
WARMUP  := 20
CONF    := 0.25
IOU     := 0.45

# ---------- YOLOv12 ----------
cpu12:
	$(BIN) --model models/yolo12n.onnx --model-name YOLOv12 \
	       --source $(VIDEO) --device cpu \
	       --output results/yolov12_cpu.mp4 \
	       --save --show \
	       --warmup $(WARMUP) --conf $(CONF) --iou $(IOU)

gpu12:
	$(BIN) --model models/yolo12n.onnx --model-name YOLOv12 \
	       --source $(VIDEO) --device cuda \
	       --output results/yolov12_gpu.mp4 \
	       --save --show \
	       --warmup $(WARMUP) --conf $(CONF) --iou $(IOU)

# ---------- YOLOv26 ----------
cpu26:
	$(BIN) --model models/yolo26n.onnx --model-name YOLOv26 \
	       --source $(VIDEO) --device cpu \
	       --output results/yolov26_cpu.mp4 \
	       --save --show \
	       --warmup $(WARMUP) --conf $(CONF) --iou $(IOU)

gpu26:
	$(BIN) --model models/yolo26n.onnx --model-name YOLOv26 \
	       --source $(VIDEO) --device cuda \
	       --output results/yolov26_gpu.mp4 \
	       --save --show \
	       --warmup $(WARMUP) --conf $(CONF) --iou $(IOU)

# ---------- Todos de una ----------
all-benchmarks: cpu12 gpu12 cpu26 gpu26

# ---------- Compilar ----------
build:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)

.PHONY: cpu12 gpu12 cpu26 gpu26 all-benchmarks build
