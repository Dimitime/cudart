# Linux
INCLUDE_PATH      = -I./include/glm
OPENGL_LIBS       = -lGLEW -lglut -lGL

CC = /usr/local/cuda/bin/nvcc
SOURCE = kernel.cu
CFLAGS = -I./include/glm -arch=sm_30
LIBS = $(OPENGL_LIBS)
EXEC = -o rt

all:
	$(CC) $(EXEC) $(LIBS) $(CFLAGS) $(SOURCE)
