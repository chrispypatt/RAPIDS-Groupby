# Makefile for GPU GroupBy Project
# EE-5351 Fall 2018
NVCC        = nvcc
NVCC_FLAGS  = -I/usr/local/cuda/include -gencode=arch=compute_60,code=\"sm_60\"
ifdef dbg
	NVCC_FLAGS  += -g -G
else
	NVCC_FLAGS  += -O3
endif

LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = groupby
OBJ	        = main.o cpuGroupby.o groupby.o HashFunc.o murmur3.o

default: $(EXE)

main.o: main.cpp cpuGroupby.h
	$(NVCC) -c -o $@ main.cpp $(NVCC_FLAGS)

HashFunc.o: murmur3.cpp murmur3.h
	$(NVCC) -c -o $@ murmur3.cpp $(NVCC_FLAGS)

HashFunc.o: HashFunc.cpp HashFunc.h
	$(NVCC) -c -o $@ HashFunc.cpp $(NVCC_FLAGS)

groupby.o: groupby.cu 
	$(NVCC) -c -o $@ groupby.cu $(NVCC_FLAGS)

cpuGroupby.o: cpuGroupby.cpp cpuGroupby.h
	$(NVCC) -c -o $@ cpuGroupby.cpp $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS) $(NVCC_FLAGS)

clean:
	rm -rf *.o $(EXE)