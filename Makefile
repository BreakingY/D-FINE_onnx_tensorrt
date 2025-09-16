# Makefile for yolo_trt_inference
CXX := g++
CXXFLAGS := -g -std=c++17 \
            -I/data/sunkx/TensorRT-8.5.1.7/include \
            -I/usr/local/cuda/include \
            -I/usr/local/opencvgpu/include/opencv4


LDFLAGS := -L/data/sunkx/TensorRT-8.5.1.7/lib \
           -L/usr/local/cuda/lib64 \
           -L/usr/local/opencvgpu/lib


LDLIBS := -lnvinfer -lcudart \
          -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui \
		  -lnppig -lnppidei -lnppial


TARGET := yolo_trt_inference
SRC := yolo_trt_inference.cpp


all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS) $(LDLIBS)


clean:
	rm -f $(TARGET)
