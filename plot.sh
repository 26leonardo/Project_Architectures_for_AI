#!/bin/bash

# Plot the output of ./k-means XX demo.txt; this script is invoked
# automatically by "make demo". It expects that there are files
# out_NNN.txt and centroids_NNN.txt containing the coordinates of data
# points and centroids at step NNN; it is assumed that both have
# dimensionality 2.
#
# Last modified on 2025-11-15 by Moreno Marzolla.

for n in `seq 0 100`; do
    OUT=`printf "out_%03d.txt" $n`
    CEN=`printf "centroids_%03d.txt" $n`
    IMG=`printf "img_%03d.png" $n`
    if [ -f "$OUT" ]; then
        {
            cat <<EOF
set term png linewidth 1.5 size 1024,768
set output "$IMG"
unset colorbox
unset xtics
unset ytics
set title "Step $n"
plot "$OUT" using 1:2:3 with p pt 6 palette notitle, \
     "$CEN" using 1:2 with p pt 2 ps 4 lt -1 lw 3 notitle
EOF
        } | gnuplot
    fi
done
