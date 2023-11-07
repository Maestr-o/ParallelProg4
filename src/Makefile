CC=g++
CFLAGS=-Wall -Wextra
BUILD=program
IFLAGS=-I/usr/local/cuda/include/
LDFLAGS=-L/usr/local/cuda/lib64/ -lOpenCL -fopenmp

all: build

build:
	@$(CC) $(IFLAGS) $(LDFLAGS) main.cpp -o $(BUILD)
