ifeq "$(HOSTNAME)" "ik-netbook-st"
NVCC_PARAM=-g -arch=sm_20 \
	-I/usr/local/cuda/include \
	-L/usr/local/cuda/lib \
	-I/home/mkotov/Projects/2012/HTM/NVIDIA_GPU_Computing_SDK/C/common/inc \
	-I/usr/include/mpi \
	-Xcompiler "-Wall"
NVCC_LD=`OcelotConfig -l` -lmpi -lmpi_cxx
else
NVCC_PARAM=-I/usr/local/cuda/include \
	-L/usr/local/cuda/lib \
	-I/usr/local/cudasdk41/C/common/inc \
	-Xcompiler "-Wall"
NVCC_LD=-lmpi -lmpi_cxx
endif

a.out: main.o experiment.o
	nvcc $(NVCC_PARAM) $(NVCC_LD) main.o experiment.o

main.o: main.cpp experiment.h
	nvcc $(NVCC_PARAM) -c main.cpp

experiment.o: experiment.cu experiment.h
	nvcc $(NVCC_PARAM) -c experiment.cu

clean:
	rm -rf *.o a.out


