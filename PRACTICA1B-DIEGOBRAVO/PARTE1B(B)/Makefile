CXX := g++
TARGET := realesrgan_video
SRC := main.cpp

OPENCV_ROOT := /home/user/librerias/opencv_final
CPPFLAGS := -I$(OPENCV_ROOT)/include/opencv4
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Wpedantic
LDFLAGS := -L$(OPENCV_ROOT)/lib -Wl,-rpath,$(OPENCV_ROOT)/lib
LDLIBS := -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -lopencv_dnn -lopencv_imgcodecs -lopencv_cudaarithm

MODEL ?= model/realesr-general-x4v3.onnx
INPUT ?= video/pajaro.mp4

.PHONY: all clean run-cpu run-gpu run-gpu-fp16

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

run-cpu: $(TARGET)
	./$(TARGET) --model "$(MODEL)" --input "$(INPUT)" --output results/output_cpu.mp4 --device cpu --tile 128 --tile-pad 10

run-gpu: $(TARGET)
	./$(TARGET) --model "$(MODEL)" --input "$(INPUT)" --output results/output_gpu.mp4 --device cuda --tile 128 --tile-pad 10

run-gpu-fp16: $(TARGET)
	./$(TARGET) --model "$(MODEL)" --input "$(INPUT)" --output results/output_gpu_fp16.mp4 --device cuda_fp16 --tile 128 --tile-pad 10

clean:
	rm -f $(TARGET)
