CXX := g++
TARGET := yolo_seg
SRC := main.cpp

OPENCV_ROOT := /home/user/librerias/opencv_final
CPPFLAGS := -I$(OPENCV_ROOT)/include/opencv4
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Wpedantic
LDFLAGS := -L$(OPENCV_ROOT)/lib -Wl,-rpath,$(OPENCV_ROOT)/lib
LDLIBS := -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_dnn -lopencv_imgcodecs -lopencv_cudaarithm

MODEL ?= modelo/best.onnx
INPUT ?= partido.mp4

.PHONY: all clean run-cpu run-gpu run-gpu-fp16

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SRC) -o $@ $(LDFLAGS) $(LDLIBS)

run-cpu: $(TARGET)
	./$(TARGET) --model "$(MODEL)" --input "$(INPUT)" --output resultado_cpu.mp4 --device cpu

run-gpu: $(TARGET)
	./$(TARGET) --model "$(MODEL)" --input "$(INPUT)" --output resultado_gpu.mp4 --device cuda

run-gpu-fp16: $(TARGET)
	./$(TARGET) --model "$(MODEL)" --input "$(INPUT)" --output resultado_gpu_fp16.mp4 --device cuda_fp16

clean:
	rm -f $(TARGET)
