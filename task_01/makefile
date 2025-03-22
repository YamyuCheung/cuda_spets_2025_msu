# Compiler options
NVCC = nvcc
CXX = g++
NVCC_FLAGS = -O3 -std=c++11 -Xcompiler -fopenmp
CXX_FLAGS = -O3 -fopenmp -std=c++11

# Set compute capability of the target GPU
# Adjust this value according to your GPU architecture
# For example: COMPUTE_CAPABILITY = 60 for Pascal architecture (GTX 1060, 1070, 1080)
#              COMPUTE_CAPABILITY = 70 for Volta architecture (V100)
#              COMPUTE_CAPABILITY = 75 for Turing architecture (RTX 2080)
#              COMPUTE_CAPABILITY = 80 for Ampere architecture (A100, RTX 3080)
#              COMPUTE_CAPABILITY = 86 for Ampere architecture (RTX 3060, 3070, 3080 Ti)
#              COMPUTE_CAPABILITY = 89 for Ada Lovelace architecture (RTX 4090)
COMPUTE_CAPABILITY ?= 86
NVCC_FLAGS += -arch=sm_$(COMPUTE_CAPABILITY)

# Define data type (double or float)
DATA_TYPE ?= double
ifeq ($(DATA_TYPE), float)
    NVCC_FLAGS += -DUSE_FLOAT
    CXX_FLAGS += -DUSE_FLOAT
endif

# Problem size (can be overridden from the command line)
L ?= 384
NVCC_FLAGS += -DL=$(L)
CXX_FLAGS += -DL=$(L)

# Target executable
TARGET = jacobi3d

# Source file
SOURCE = jacobi3d.cu

# Target
all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

clean:
	rm -f $(TARGET)

run_cpu: $(TARGET)
	./$(TARGET) --cpu

run_gpu: $(TARGET)
	./$(TARGET) --gpu

compare: $(TARGET)
	./$(TARGET) --compare

test_float: clean
	$(MAKE) DATA_TYPE=float
	./$(TARGET) --compare

test_double: clean
	$(MAKE) DATA_TYPE=double
	./$(TARGET) --compare

# Test with different problem sizes
test_sizes: clean
	@echo "Testing with different problem sizes..."
	$(MAKE) L=128 
	./$(TARGET) --compare
	$(MAKE) clean
	$(MAKE) L=256
	./$(TARGET) --compare
	$(MAKE) clean
	$(MAKE) L=384
	./$(TARGET) --compare

.PHONY: all clean run_cpu run_gpu compare test_float test_double test_sizes
