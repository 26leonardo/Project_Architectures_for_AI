/****************************************************************************
 *
 * inputgen.c - Generate random input for the K-Means algorithm.
 *
 * Copyright (C) 2025 Moreno Marzolla <https://unibo.it/sitoweb/moreno.marzolla/>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * --------------------------------------------------------------------------
 *
 * This program generates a random input for the K-Means algorithm.
 *
 * To compile:
 *
 * gcc -std=c99 -Wall -Wpedantic inputgen.c -o inputgen
 *
 * To execute:
 *
 * ./inputgen points_per_cluster n_dims n_clusters
 *
 * The program generates `n_clusters` hyper-rectangles in `n_dims`
 * dimensions with random position; each hyper-rectangle contains
 * `points_per_cluster` random points. Therefore, the total number of
 * points is` points_per_cluster * n_clusters`.
 *
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

double rand01( void )
{
    return rand() / (double)RAND_MAX;
}

double randab( double a, double b )
{
    return a + rand01() * (b-a);
}

/* Generate `n` random points inside a `D`-dimensional box with center
   `center` and side `2*r`. */
void gen_points( float *center, int D, float r, int n )
{
    for (int i=0; i<n; i++) {
        for (int d=0; d<D; d++) {
            printf("%f ", center[d] + randab(-r, r));
        }
        printf("\n");
    }
}

void init_center( float *p, int D )
{
    for (int d=0; d<D; d++) {
        p[d] = randab(0, 200);
    }
}

int main( int argc, char *argv[] )
{
    int points_per_cluster, n_dims, n_clusters;

    if (argc != 4) {
        fprintf(stderr, "Usage: %s points_per_cluster n_dims n_clusters\n", argv[0]);
        return EXIT_FAILURE;
    }

    srand(17); /* Deterministic initialization of the PRNG. */

    points_per_cluster = atoi(argv[1]);
    n_dims = atoi(argv[2]);
    n_clusters = atoi(argv[3]);

    float *center = (float*)malloc(n_dims * sizeof(*center));

    for (int c=0; c<n_clusters; c++) {
        const double side = randab(10, 30);
        init_center(center, n_dims);
        gen_points(center, n_dims, side, points_per_cluster);
    }

    free(center);
    return EXIT_SUCCESS;
}
