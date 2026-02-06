CC      = gcc
NVCC    = nvcc

CFLAGS  = -std=c99 -Wall -Wpedantic -O3
OMPFLAGS= -fopenmp
CUDAFLAGS = -O3 -arch=sm_80 # ADJUST ARCHITECTURE AS NEEDED

LDFLAGS = -lm

SRC_DIR = src
UTILS_DIR = utils
DATA_DIR = data

SERIAL = k-means
OMP    = omp-k-means
CUDA   = cuda-k-means
INPUTGEN = inputgen

.PHONY: all clean distclean demo

all: $(SERIAL) $(OMP) $(CUDA) $(INPUTGEN)

$(SERIAL):
	$(CC) $(CFLAGS) $(SRC_DIR)/k-means.c -o $@ $(LDFLAGS)

$(OMP):
	$(CC) $(CFLAGS) $(OMPFLAGS) $(SRC_DIR)/omp-k-means.c -o $@ $(LDFLAGS)

$(CUDA):
	$(NVCC) $(CUDAFLAGS) $(SRC_DIR)/cuda-k-means.c -o $@ $(LDFLAGS)

$(INPUTGEN):
	$(CC) $(CFLAGS) $(UTILS_DIR)/inputgen.c -o $@

demo: $(SERIAL)
	rm -f centroids_*.txt out_*.txt img_*.png demo.avi
	$(CC) $(CFLAGS) -DMAKE_MOVIE $(SRC_DIR)/k-means.c -o $(SERIAL) $(LDFLAGS)
	./$(SERIAL) 5 $(DATA_DIR)/demo.txt demo.out
	rm -f demo.out
	$(UTILS_DIR)/plot.sh
	ffmpeg -pattern_type glob -stream_loop 5 -y -r 1 -i "img_*.png" -vcodec mpeg4 -r 1 demo.avi

clean:
	rm -f $(SERIAL) $(OMP) $(CUDA) $(INPUTGEN) *.o

distclean: clean
	rm -f *~ centroids_*.txt out_*.txt img_*.png *.avi demo.out