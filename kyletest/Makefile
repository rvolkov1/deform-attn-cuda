# ---- Compiler settings (MATCH dat_cuda) ----
NVCC     := nvcc
ARCH     := -arch=sm_75
PTX_OPTS := -Xptxas -O3 -O3

# ---- Paths ----
SRC      := test_proj_q.cu
OBJ      := test_proj_q.o
TARGET   := test_proj_q

# ---- Rules ----
all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) $(ARCH) $(PTX_OPTS) $(OBJ) -o $(TARGET)

$(OBJ): $(SRC)
	$(NVCC) $(ARCH) $(PTX_OPTS) -c $(SRC) -o $(OBJ)

clean:
	rm -f $(OBJ) $(TARGET)

run: all
	./$(TARGET)
