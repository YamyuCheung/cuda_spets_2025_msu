# Makefile for compiling and submitting single-GPU and dual-GPU CUDA jobs on LSF
# Variables for common settings
NVCC = nvcc
ARCH = sm_60
CFLAGS = -Xcompiler -fopenmp -O3

# Variables for single-GPU version
TARGET_SINGLE = adi_program
SRC_SINGLE = main.cu
LSF_OUT_SINGLE = adi_report_polus.txt

# Variables for dual-GPU version
TARGET_DUAL = adi_program_dual
SRC_DUAL = main_dual_gpus.cu
LSF_OUT_DUAL = adi_report_polus_dual_gpus_version.txt

# Default target: compile and submit both jobs
all: run_single run_dual

# Compile targets
$(TARGET_SINGLE): $(SRC_SINGLE)
	$(NVCC) -arch=$(ARCH) $(CFLAGS) -o $(TARGET_SINGLE) $(SRC_SINGLE)

$(TARGET_DUAL): $(SRC_DUAL)
	$(NVCC) -arch=$(ARCH) $(CFLAGS) -o $(TARGET_DUAL) $(SRC_DUAL)

# Build both executables
build: $(TARGET_SINGLE) $(TARGET_DUAL)

# Submit single-GPU job
run_single: $(TARGET_SINGLE)
	bsub -gpu "num=1:mode=exclusive_process" -o $(LSF_OUT_SINGLE) ./$(TARGET_SINGLE)

# Submit dual-GPU job
run_dual: $(TARGET_DUAL)
	bsub -gpu "num=2:mode=shared:mps=yes" -o $(LSF_OUT_DUAL) ./$(TARGET_DUAL)

# Clean build and output files
clean:
	rm -f $(TARGET_SINGLE) $(TARGET_DUAL) $(LSF_OUT_SINGLE) $(LSF_OUT_DUAL)

# Just compile without submitting jobs
compile: $(TARGET_SINGLE) $(TARGET_DUAL)
