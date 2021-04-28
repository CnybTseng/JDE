CROSS :=aarch64-linux-gnu-
#CC :=$(CROSS)g++
CC :=g++

ROOT_DIR := $(PWD)/..
# ROOT_DIR := .
SRC_DIR := $(ROOT_DIR)/mot
INC_DIR := $(ROOT_DIR)/include
INC_DIR += -I$(ROOT_DIR)/include/opencv
INC_DIR += -I$(ROOT_DIR)/include/ffmpeg
INC_DIR += -I$(ROOT_DIR)/include/mot
LIB_DIR := $(ROOT_DIR)/lib
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS  := $(SRCS:%.cpp=%.o)

ALL_LIBS := -lalgfwsdk -lcurl -levent -levent_pthreads -lmosquitto -lssl -lcrypto -lpthread -lm
# opencv
ALL_LIBS += -lopencv_calib3d -lopencv_core -lopencv_features2d
ALL_LIBS += -lopencv_flann -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc
ALL_LIBS += -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching
ALL_LIBS += -lopencv_video -lopencv_videoio

# ffmpeg
ALL_LIBS += -lavcodec -lavdevice -lavfilter -lavformat -lavutil
ALL_LIBS += -lswresample -lswscale

# yuv
ALL_LIBS += -lyuv

# aarch64_mlu220   
ALL_LIBS += -lcncodec -lcnrt -lcndrv

# multi-object tracking
ALL_LIBS += -lmot

CFLAGS += -Wall
CFLAGS += -I$(SRC_DIR)
CFLAGS += -I$(INC_DIR)
CFLAGS += -L$(LIB_DIR)
CFLAGS += -g
CFLAGS += -std=gnu++11 -Wl,-rpath=$(LIB_DIR)

TARGET := mot


all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(ALL_LIBS) $(CFLAGS)

$(OBJS):%.o:%.cpp
	$(CC) -o $@ -c $< $(CFLAGS)

clean:
	@rm -f $(TARGET)
	@rm -f $(OBJS)
