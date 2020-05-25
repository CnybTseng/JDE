ARM ?= 0
VULKAN ?= 1
MODEL_PATH ?= model/darknet53/lightly
GDB ?= 0

ifeq ($(ARM), 1)
    CXX = aarch64-himix100-linux-g++
    RELEASE = build-aarch64-himix100-linux-gnu/release
else
    CXX = /opt/rh/devtoolset-6/root/usr/bin/g++
    RELEASE = build/release
endif

THIRDPARTY_LIBS := thirdparty/ncnn/$(RELEASE)/lib/libncnn.a

THIRDPARTY_LIBS += -L/usr/local/lib64

ifeq ($(VULKAN), 1)
    THIRDPARTY_LIBS += -Lthirdparty/1.1.114.0/x86_64/lib
endif

LDFLAGS := `/usr/local/bin/python3-config --ldflags`

LIBS := $(THIRDPARTY_LIBS)
LIBS += -lpthread -lm -ldl

ifeq ($(VULKAN), 1)
    LIBS += -lvulkan
endif

LIBS += -lopencv_core -lopencv_video -lopencv_imgcodecs -lopencv_imgproc

CXXFLAGS := -Isrc
CXXFLAGS += -Isrc/core
CXXFLAGS += -I$(MODEL_PATH)
CXXFLAGS += -I/usr/local/include/opencv4
CXXFLAGS += -I/usr/local/include/python3.6m
CXXFLAGS += -I/home/image/tseng/venv/lib/python3.6/site-packages/numpy/core/include/numpy
CXXFLAGS += -Ithirdparty/ncnn/$(RELEASE)/include/ncnn/
CXXFLAGS += -Ithirdparty/libjpeg-turbo/$(RELEASE)/include
CXXFLAGS += -Ithirdparty/pthreadpool/$(RELEASE)/include
CXXFLAGS += -Ithirdparty/yaml-cpp/$(RELEASE)/include
CXXFLAGS += -fopenmp
CXXFLAGS += -fpermissive
CXXFLAGS += -fPIC
CXXFLAGS += -fvisibility=hidden
CXXFLAGS += -fvisibility-inlines-hidden
CXXFLAGS += -Wall
CXXFLAGS += -O3

CXXFLAGS += -DMOT_DLL
CXXFLAGS += -DPTHREADPOOL_NO_DEPRECATED_API

ifeq ($(VULKAN), 1)
    CXXFLAGS += -Ithirdparty/1.1.114.0/x86_64/include
    CXXFLAGS += -DNCNN_VULKAN
endif

CXXFLAGS += `/usr/local/bin/python3-config --cflags`

ifeq ($(GDB), 1)
    CXXFLAGS += -g
endif

SRCS := $(wildcard *.cpp)
SRCS += $(wildcard ./src/*.cpp)
SRCS += $(wildcard ./src/core/*.cpp)
TARGET := mot

OBJS := $(SRCS:%.cpp=%.o)
EXE_OBJS := mot.o
LIB_OBJS := $(filter-out $(EXE_OBJS),$(OBJS))
SLIB = libmot.so

RPATH := -Wl,-rpath=./
ifeq ($(VULKAN), 1)
    RPATH += -Wl,-rpath=thirdparty/1.1.114.0/x86_64/lib
endif
RPATH += -Wl,-rpath=/usr/local/lib64

.PHONY : clean all

all: $(SLIB) $(TARGET)

mot:$(SLIB) mot.o
	@$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) -L. -lmot $(LIBS) $(RPATH)

$(SLIB):$(LIB_OBJS)
	@$(CXX) $(CXXFLAGS) -DMOT_DLL_EXPORTS -shared -Wl,-soname=libmot.so -o $@ $^ $(LIBS)

clean:
	@rm -f $(TARGET)
	@rm -f $(OBJS)
	@rm -f $(SLIB)