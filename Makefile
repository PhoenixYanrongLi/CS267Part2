# Load CUDA using the following command
# module load cuda

#
# Stampede
#
CC = nvcc
MPCC = nvcc
OPENMP = 
CFLAGS = -O3 -arch=sm_35
NVCCFLAGS = -O3 -arch=sm_35
LIBS = -lm

TARGETS = serial gpu gpu_new

all:	$(TARGETS)

serial: serial.o common.o
	$(CC) -o $@ $(LIBS) serial.o common.o
gpu: gpu.o common.o
	$(CC) -o $@ $(NVCCLIBS) gpu.o common.o
gpu_new: gpu_new.o common.o
	$(CC) -o $@ $(NVCCLIBS) gpu_new.o common.o

serial.o: serial.cu common.h
	$(CC) -c $(CFLAGS) serial.cu
gpu_new.o: gpu_new.cu common.h
	$(CC) -c $(NVCCFLAGS) gpu_new.cu
gpu.o: gpu.cu common.h
	$(CC) -c $(NVCCFLAGS) gpu.cu
common.o: common.cu common.h
	$(CC) -c $(CFLAGS) common.cu

clean:
	rm -f *.o $(TARGETS)
