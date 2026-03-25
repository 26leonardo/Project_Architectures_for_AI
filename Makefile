CC      = gcc
NVCC    = nvcc

CFLAGS  = -std=c99 -Wall -Wpedantic 
O3 		= -O3
OMPFLAGS= -fopenmp
MAXITERS= -DMAX_ITER_FIXED=150
# RTX 4060 Ti is Ada Lovelace → sm_89. Adjust if compiling on a different GPU.
CUDAFLAGS = -arch=sm_89 # ADJUST ARCHITECTURE AS NEEDED #-O3

LDFLAGS = -lm

SRC_DIR   = src
UTILS_DIR = utils
DATA_DIR  = data

SERIAL   = k-means
OMP      = omp-k-means
OMP_V2   = omp-k-means-v2
OMP_V3   = omp-k-means-v3
OMP_V4   = omp-k-means-v4
OMP_O3   = omp-k-means-o3
OMP_O3_V4   = omp-k-means-o3-v4
CUDA     = cuda-k-means
INPUTGEN = inputgen

.PHONY: all clean distclean demo

all: $(OMP) $(OMP_V2) $(OMP_V3) $(OMP_V4) $(OMP_O3) $(OMP_O3_V4) $(INPUTGEN) $(CUDA)# $(SERIAL) 

# Serial baseline: omp-k-means.c compiled with OpenMP but run with
# OMP_NUM_THREADS=1. This is the CORRECT baseline for speedup measurement:
# a parallel program with p=1 avoids comparing against a structurally
# different serial program.
$(SERIAL):
	$(CC) $(CFLAGS) $(OMPFLAGS) $(SRC_DIR)/omp-k-means.c -o $@ $(LDFLAGS)

$(OMP):
	$(CC) $(MAXITERS) $(CFLAGS) $(OMPFLAGS) $(SRC_DIR)/omp-k-means.c -o $@ $(LDFLAGS)
$(OMP_O3):
	$(CC) $(MAXITERS) $(CFLAGS) $(O3) $(OMPFLAGS) $(SRC_DIR)/omp-k-means.c -o $@ $(LDFLAGS)
$(OMP_V2):
	$(CC) $(MAXITERS) $(CFLAGS) $(OMPFLAGS) $(SRC_DIR)/new/omp-k-means-v2.c -o $@ $(LDFLAGS)
$(OMP_V3):
	$(CC) $(MAXITERS) $(CFLAGS) $(OMPFLAGS) $(SRC_DIR)/new/omp-k-means-v3.c -o $@ $(LDFLAGS)
$(OMP_V4):
	$(CC) $(MAXITERS) $(CFLAGS) $(OMPFLAGS) $(SRC_DIR)/new/omp-k-means-v4.c -o $@ $(LDFLAGS)
$(OMP_O3_V4):
	$(CC) $(MAXITERS) $(CFLAGS) $(O3) $(OMPFLAGS) $(SRC_DIR)/new/omp-k-means-v4.c -o $@ $(LDFLAGS)

# Source is .cu (nvcc requires CUDA source extension).
$(CUDA):
	$(NVCC) $(MAXITERS) $(CUDAFLAGS) $(SRC_DIR)/cuda-k-means.cu -o $@ $(LDFLAGS)

$(INPUTGEN):
	$(CC) $(CFLAGS) $(UTILS_DIR)/inputgen.c -o $@

demo: $(SERIAL)
	rm -f temp/centroids_*.txt temp/out_*.txt img/img_*.png demo.avi
	$(CC) $(CFLAGS) $(OMPFLAGS) -DMAKE_MOVIE $(SRC_DIR)/omp-k-means.c -o $(SERIAL) $(LDFLAGS)
	OMP_NUM_THREADS=1 ./$(SERIAL) 5 $(DATA_DIR)/demo.txt demo.out
	rm -f demo.out
	$(UTILS_DIR)/plot.sh
	ffmpeg -pattern_type glob -stream_loop 5 -y -r 1 -i "img/img_*.png" -vcodec mpeg4 -r 1 demo.avi

clean:
	rm -f $(OMP) $(OMP_V2) $(OMP_V3) $(OMP_V4) $(OMP_O3) $(OMP_O3_V4) $(CUDA) $(INPUTGEN) *.o

distclean: clean
	rm -f *~ temp/centroids_*.txt temp/out_*.txt img/img_*.png *.avi demo.out