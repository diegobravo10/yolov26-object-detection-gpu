CXX := g++
TARGET := realesrgan_bench
SRC := main.cpp

OPENCV_ROOT := /home/ariel/opencv-cuda-install
CPPFLAGS := -I$(OPENCV_ROOT)/include/opencv4
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Wpedantic
LDFLAGS := -L$(OPENCV_ROOT)/lib -Wl,-rpath,$(OPENCV_ROOT)/lib:/home/ariel/cuda-local/usr/lib/x86_64-linux-gnu
LDLIBS := -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_dnn -lopencv_imgcodecs -lopencv_cudaarithm

MODEL ?= modelo/realesr-general-x4v3.onnx
INPUT ?= videos/dbz_270p.mp4
TILE ?= 128
TILE_PAD ?= 10

.PHONY: all clean run-cpu run-gpu run-gpu-fp16

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

run-cpu: $(TARGET)
	./$(TARGET) --model "$(MODEL)" --input "$(INPUT)" \
	            --output resultados/resultado_cpu.mp4 \
	            --device cpu --tile $(TILE) --tile-pad $(TILE_PAD) --show true --save true

run-gpu: $(TARGET)
	./$(TARGET) --model "$(MODEL)" --input "$(INPUT)" \
	            --output resultados/resultado_gpu.mp4 \
	            --device cuda --tile $(TILE) --tile-pad $(TILE_PAD) --show true --save true

run-gpu-fp16: $(TARGET)
	./$(TARGET) --model "$(MODEL)" --input "$(INPUT)" \
	            --output resultados/resultado_gpu_fp16.mp4 \
	            --device cuda_fp16 --tile $(TILE) --tile-pad $(TILE_PAD) --show true --save true

clean:
	rm -f $(TARGET)
	rm -f resultados/*.mp4 resultados/*.csv
