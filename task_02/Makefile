# Makefile for compiling and submitting a CUDA job on LSF using 2 GPUs

# Variables
NVCC = nvcc
ARCH = sm_60
CFLAGS = -Xcompiler -fopenmp -O3
TARGET = myprogram
SRC = main.cu
LSF_OUT = adi_report_polus.txt

# Default target: compile and submit the job
all: $(TARGET)
	bsub -gpu "num=2:mode=shared:mps=yes" -o $(LSF_OUT) ./$(TARGET)

# Compile target
$(TARGET): $(SRC)
	$(NVCC) -arch=$(ARCH) $(CFLAGS) -o $(TARGET) $(SRC)

# Clean build and output files
clean:
	rm -f $(TARGET) $(LSF_OUT)

# Submit only (without compiling)
run:
	bsub -gpu "num=2:mode=shared:mps=yes" -o $(LSF_OUT) ./$(TARGET)