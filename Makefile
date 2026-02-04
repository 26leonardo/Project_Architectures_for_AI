CFLAGS+=-std=c99 -Wall -Wpedantic
EXES:=k-means inputgen

.PHONY: demo clean distclean

ALL: $(EXES)

demo:
	$(CC) $(CFLAGS) -DMAKE_MOVIE k-means.c -o k-means
	\rm -f centroids_*.txt out_*.txt img_*.png
	./k-means 5 demo.txt demo.out
	\rm demo.out
	./plot.sh
	ffmpeg -pattern_type glob -stream_loop 5 -y -r 1 -i "img_*.png" -vcodec mpeg4 -r 1 demo.avi

demo.txt: inputgen
	./inputgen 20 2 50 > $@

clean:
	\rm -f $(EXES) *.o

distclean: clean
	\rm -f *~ centroids_*.txt out_*.txt img_*.png *.avi demo.out
