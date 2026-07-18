OPENCV_INC=/home/ariel/opencv-cuda-install/include/opencv4
OPENCV_LIB=/home/ariel/opencv-cuda-install/lib

CXX=g++
CXXFLAGS=-std=c++17 -O2 -Wall -I$(OPENCV_INC)
LDFLAGS=-L$(OPENCV_LIB) -Wl,-rpath,$(OPENCV_LIB):/home/ariel/cuda-local/usr/lib/x86_64-linux-gnu

LIBS=\
	-lopencv_core \
	-lopencv_highgui \
	-lopencv_imgproc \
	-lopencv_imgcodecs \
	-lopencv_video \
	-lopencv_videoio \
	-lopencv_cudaimgproc \
	-lopencv_cudafilters \
	-lopencv_cudaarithm \
	-pthread

SRC=main.cpp
BIN=parte1c

.PHONY: all clean run run-video check

all: $(BIN)

$(BIN): $(SRC)
	$(CXX) $(SRC) $(CXXFLAGS) $(LDFLAGS) $(LIBS) -o $(BIN)

run: $(BIN)
	./$(BIN) 0

run-video: $(BIN)
	./$(BIN) $(VIDEO)

clean:
	rm -f $(BIN)
	rm -f resultados/resultados_parte_1c.csv
	rm -f resultados/resumen_parte_1c.txt
	rm -f resultados/captura_frame_*.png

check:
	ls $(OPENCV_LIB) | grep cuda
