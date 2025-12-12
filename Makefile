# ---- Compiler settings ----
NVCC        := nvcc
ARCH        := -arch=sm_75
PTX_OPTS    := -Xptxas -O3 -O3

BASE_DIR    := dat_cuda

# ---- Sources ----
CU_SRC      := $(BASE_DIR)/main.cu

# ---- Objects ----
OBJS        := $(BASE_DIR)/main.o

# ---- Target ----
TARGET      := $(BASE_DIR)/exe

# ---- Rules ----
all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(ARCH) $(PTX_OPTS) $(OBJS) -o $(TARGET)

$(BASE_DIR)/main.o: $(BASE_DIR)/main.cu $(BASE_DIR)/read_utils.h
	$(NVCC) $(ARCH) $(PTX_OPTS) -c $(BASE_DIR)/main.cu -o $(BASE_DIR)/main.o

clean:
	rm -f $(BASE_DIR)/*.o $(TARGET)

run: all
	./$(TARGET)