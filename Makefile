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
	$(CC) $(CFLAGS) $(OMPFLAGS) $(SRC_DIR)/omp-k-means.c -o $@ $(LDFLAGS)

$(OMP):
	$(CC) $(CFLAGS) $(OMPFLAGS) $(SRC_DIR)/omp-k-means.c -o $@ $(LDFLAGS)

$(CUDA):
	$(NVCC) $(CUDAFLAGS) $(SRC_DIR)/cuda-k-means.c -o $@ $(LDFLAGS)

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
	rm -f $(SERIAL) $(OMP) $(CUDA) $(INPUTGEN) *.o

distclean: clean
	rm -f *~ temp/centroids_*.txt temp/out_*.txt img/img_*.png *.avi demo.out