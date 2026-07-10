OPENCV_INC=/home/user/librerias/opencv_final/include/opencv4
OPENCV_LIB=/home/user/librerias/opencv_final/lib

CXX=g++
CXXFLAGS=-std=c++17 -I$(OPENCV_INC)
LDFLAGS=-L$(OPENCV_LIB) -Wl,-rpath,$(OPENCV_LIB)

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
BIN=parte1c.bin

all:
	$(CXX) $(SRC) $(CXXFLAGS) $(LDFLAGS) $(LIBS) -o $(BIN)

run:
	./$(BIN)

run-video:
	./$(BIN) $(VIDEO)

clean:
	rm -f $(BIN)
	rm -f resultados_parte_1c.csv
	rm -f resumen_parte_1c.txt
	rm -f captura_parte_1c_frame_*.png

check:
	ls $(OPENCV_LIB) | grep cuda