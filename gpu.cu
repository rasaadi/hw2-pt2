#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include "common.h"

#define NUM_THREADS 256
#define density 0.0005
#define num_particles_per_bin 500

extern double size;
//
//  benchmarking program
//

typedef struct
{
    int particles[num_particles_per_bin];
    int length;
} sampleBin;


__device__ void apply_force_gpu(particle_t &particle, particle_t &neighbor)
{
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if( r2 > cutoff*cutoff )
        return;
    //r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
    double r = sqrt( r2 );

    //
    //  very simple short-range repulsive force
    //
    double coef = ( 1 - cutoff / r ) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;

}

__global__ void compute_forces_gpu(particle_t * particles, sampleBin* bins, int n_rows, int n_cols, double bin_side, int n)
{
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= n) return;
    particles[tid].ax = particles[tid].ay = 0;

    int row = floor(particles[tid].x / bin_side);
    int col = floor(particles[tid].y / bin_side);

    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            int bin_ind_x = row + i;
            int bin_ind_y = col + j;
            if (bin_ind_x >= 0 && bin_ind_x < n_rows && bin_ind_y >= 0 && bin_ind_y < n_cols) {
                int neighbor_ind = bin_ind_x*n_cols + bin_ind_y;
                sampleBin& neighbor_bin = bins[neighbor_ind];
                for (int z = 0; z<neighbor_bin.length; z++) {
                    particle_t& neighbor = particles[neighbor_bin.particles[z]];
                    apply_force_gpu(particles[tid], neighbor);
                }
            }
        }
    }
}

__global__ void move_gpu (particle_t * particles, int n, double size)
{

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= n) return;

    particle_t * p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x  += p->vx * dt;
    p->y  += p->vy * dt;

    while( p->x < 0 || p->x > size )
    {
        p->x  = p->x < 0 ? -(p->x) : 2*size-p->x;
        p->vx = -(p->vx);
    }
    while( p->y < 0 || p->y > size )
    {
        p->y  = p->y < 0 ? -(p->y) : 2*size-p->y;
        p->vy = -(p->vy);
    }

}

__global__ void reassign_bins (particle_t* particles, sampleBin* bins, int roll, int num_bins, double bin_side, int n_cols) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= num_bins) return;

    sampleBin &bin = bins[tid];
    bin.length = 0;
    for (int i = 0; i < roll; i++) {
        int x_ind = floor(particles[i].x / bin_side);
        int y_ind = floor(particles[i].y / bin_side);
        if (x_ind * n_cols + y_ind == tid) {
            bin.particles[bin.length] = i;
            bin.length += 1;
        }
    }
}


int main( int argc, char **argv )
{
    // This takes a few seconds to initialize the runtime
    cudaThreadSynchronize();

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        return 0;
    }

    int n = read_int( argc, argv, "-n", 1000 );

    char *savename = read_string( argc, argv, "-o", NULL );

    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );

    // GPU particle data structure
    particle_t * d_particles;
    cudaMalloc((void **) &d_particles, n * sizeof(particle_t));

    set_size( n );

    init_particles( n, particles );

    cudaThreadSynchronize();
    double copy_time = read_timer( );

    // Copy the particles to the GPU
    cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);

    int n_rows = ceil(sqrt(NUM_THREADS));
    int n_cols = n_rows;
    int cross = n_rows * n_cols;
    double bin_side = size / n_rows;

    sampleBin* cpu_bins = (Bin* ) malloc(cross * sizeof(Bin));
    for (int i = 0; i < cross; i++) {
        cpu_bins[i] = sampleBin();
        cpu_bins[i].length = 0;
        for (int j = 0; j< num_particles_per_bin; j++) {
            cpu_bins[i].particles[j] = -1;
        }
    }

    for ( int i = 0; i < n; i++) {
        int x_ind = floor(particles[i].x / bin_side);
        int y_ind = floor(particles[i].y / bin_side);
        cpu_bins[x_ind*n_cols + y_ind].particles[cpu_bins[x_ind*n_cols + y_ind].length] = i;
        cpu_bins[x_ind*n_cols + y_ind].length += 1;
    }

    sampleBin* gpu_bins;
    cudaMalloc((void **) &gpu_bins, cross * sizeof(sampleBin));

    cudaMemcpy(gpu_bins, cpu_bins, cross * sizeof(sampleBin), cudaMemcpyHostToDevice);

    cudaThreadSynchronize();
    copy_time = read_timer( ) - copy_time;

    //
    //  simulate a number of time steps
    //
    cudaThreadSynchronize();
    double simulation_time = read_timer( );
    int blocks = (n + NUM_THREADS - 1) / NUM_THREADS;
    int sqrt_num_threads = floor(sqrt(NUM_THREADS));
    int bin_blks = (cross + sqrt_num_threads - 1 ) / sqrt_num_threads;
    for( int step = 0; step < NSTEPS; step++ )
    {
        //
        //  compute forces
        //


        compute_forces_gpu << blocks, NUM_THREADS >> (d_particles, gpu_bins, n_rows, n_cols, bin_side, n);
        //
        //  move particles
        //
        move_gpu << blocks, NUM_THREADS >> (d_particles, n, size);
        cudaThreadSynchronize();


        if (step % 4 == 0) {
            cudaError_t error = cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
            if (error != cudaSuccess) {
                printf("1. %s \n", cudaGetErrorString(error));
                exit(1);
            }
            for (int i = 0; i < cross; i++) {
                cpu_bins[i].length = 0;
            }

            for ( int i = 0; i < n; i++) {
                int x_ind = floor(particles[i].x / bin_side);
                int y_ind = floor(particles[i].y / bin_side);
                cpu_bins[x_ind*n_cols + y_ind].particles[cpu_bins[x_ind*n_cols + y_ind].length] = i;
                cpu_bins[x_ind*n_cols + y_ind].length += 1;
            }
            cudaError_t error1 = cudaMemcpy(gpu_bins, cpu_bins, cross * sizeof(sampleBin), cudaMemcpyHostToDevice);
            if (error1 != cudaSuccess) {
                printf("2. %s \n", cudaGetErrorString(error1));
                exit(1);
            }
        }

        //
        //  save if necessary
        //
        if( fsave && (step%SAVEFREQ) == 0 ) {
            cudaThreadSynchronize();
            // Copy the particles back to the CPU
            cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
            save( fsave, n, particles);
        }
    }
    cudaThreadSynchronize();
    simulation_time = read_timer( ) - simulation_time;

    printf( "CPU-GPU copy time = %g seconds\n", copy_time);
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );

    free( particles );
    cudaFree(d_particles);
    if( fsave )
        fclose( fsave );

    return 0;
}

