CXX := g++
TARGET := yolo_bench
SRC := main.cpp

OPENCV_ROOT := /home/ariel/opencv-cuda-install
CPPFLAGS := -I$(OPENCV_ROOT)/include/opencv4
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Wpedantic
LDFLAGS := -L$(OPENCV_ROOT)/lib -Wl,-rpath,$(OPENCV_ROOT)/lib:/home/ariel/cuda-local/usr/lib/x86_64-linux-gnu
LDLIBS := -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_dnn -lopencv_cudaarithm

MODEL_DIR := modelos
VIDEO_DIR := videos
CONFIG_DIR := config
RESULTS_DIR := resultados

MODEL_12 ?= $(MODEL_DIR)/yolo12n.onnx
MODEL_26 ?= $(MODEL_DIR)/yolo26n.onnx
VIDEO ?= $(VIDEO_DIR)/Los_Angeles.mp4
CLASSES ?= $(CONFIG_DIR)/classes.txt
WARMUP := 20
CONF := 0.25
IOU := 0.45

.PHONY: all clean build cpu12 gpu12 cpu26 gpu26 all-benchmarks

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

build: $(TARGET)

# ---------- YOLOv12 ----------
cpu12: $(TARGET)
	./$(TARGET) --model "$(MODEL_12)" --model-name YOLOv12 \
	           --source "$(VIDEO)" --device cpu \
	           --output "$(RESULTS_DIR)/yolov12_cpu.mp4" \
	           --classes "$(CLASSES)" \
	           --save true --show true \
	           --warmup $(WARMUP) --conf $(CONF) --iou $(IOU)

gpu12: $(TARGET)
	./$(TARGET) --model "$(MODEL_12)" --model-name YOLOv12 \
	           --source "$(VIDEO)" --device cuda \
	           --output "$(RESULTS_DIR)/yolov12_gpu.mp4" \
	           --classes "$(CLASSES)" \
	           --save true --show true \
	           --warmup $(WARMUP) --conf $(CONF) --iou $(IOU)

# ---------- YOLOv26 ----------
cpu26: $(TARGET)
	./$(TARGET) --model "$(MODEL_26)" --model-name YOLOv26 \
	           --source "$(VIDEO)" --device cpu \
	           --output "$(RESULTS_DIR)/yolov26_cpu.mp4" \
	           --classes "$(CLASSES)" \
	           --save true --show true \
	           --warmup $(WARMUP) --conf $(CONF) --iou $(IOU)

gpu26: $(TARGET)
	./$(TARGET) --model "$(MODEL_26)" --model-name YOLOv26 \
	           --source "$(VIDEO)" --device cuda \
	           --output "$(RESULTS_DIR)/yolov26_gpu.mp4" \
	           --classes "$(CLASSES)" \
	           --save true --show true \
	           --warmup $(WARMUP) --conf $(CONF) --iou $(IOU)

# ---------- Todos ----------
all-benchmarks: cpu12 gpu12 cpu26 gpu26

clean:
	rm -f $(TARGET)
	rm -f $(RESULTS_DIR)/*.mp4 $(RESULTS_DIR)/*.csv $(RESULTS_DIR)/*.md
