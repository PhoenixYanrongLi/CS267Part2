#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include "common.h"

#define NUM_THREADS 256

#define density 0.0005
#define mass    0.01
#define cutoff  0.01
#define min_r   (cut_off/100)
#define dt      0.0005
#define binParticleMax  20

using namespace std;

extern double size;
//
//  benchmarking program
//

struct direction
{
    int x_change;
    int y_change;
};

direction Directions[] = {{-1,-1},{0,-1},{1,-1},{-1,0},{0,0},{1,0},{-1,1},{0,1},{1,1}};


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

__global__ void distribute_particles_gpu (particle_t* particles, particle_t** bins, 
                                   int binNumPerEdge, int* binParticleNum, int n) {
    //get rid of the outer for loop
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    int bin_x = floor(particles[tid].x / cutoff);
    int bin_y = floor(particles[tid].y / cutoff);
    int binIndex = bin_x + bin_y * binNumPerEdge;
    // increase binParticleNum atomically
    int particleIdxInBin = atomicAdd(binParticleNum+binIndex, 1); 
    // set this pointer in the bins to the particle
    bins[binParticleMax * binIndex + particleIdxInBin] = particles + tid; 
}

__global__ void apply_force_per_bin_gpu (particle_t** bins, int binNumPerEdge, int* binParticleNum) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= binNumPerEdge * binNumPerEdge) return;

    for (int i = 0; i < binParticleNum[tid]; i++) {
      apply_force_per_particle_gpu(*(bins[tid * binParticleMax + i]), bins, tid, binNumPerEdge, binParticleNum);
    }
}

__device__ void apply_force_particle_bin_gpu (particle_t &particle, particle_t** bins, int binIdx,
                                              int binNumPerEdge, int* binParticleNum) {
    particle.ax = particle.ay = 0;

    int bin_x = binIdx % binNumPerEdge;
    int bin_y = binIdx / binNumPerEdge;

    for(int i = 0 ; i < 9; i++)
    {
        int bin_x_new = bin_x + Directions[i].x_change;
        int bin_y_new = bin_y + Directions[i].y_change;

        if(bin_x_new >= 0 && bin_x_new < binNumPerEdge && bin_y_new >= 0 && bin_y_new < binNumPerEdge)
        {
            int binIdx_new = bin_x_new + bin_y_new * binNumPerEdge;
            for(int j = 0; j < binParticleNum[binIdx_new]; j++)
            {
                apply_force_gpu(particle, *(bins[binIdx_new * binParticleMax + j]));
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

    //
    //  bounce from walls
    //
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
        printf( "-s <filename> to specify a summary file name\n" );
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );

    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );
    
    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname ? fopen ( sumname, "a" ) : NULL;
    particle_t * particles = (particle_t*) malloc( n * sizeof(particle_t) );

    // GPU particle data structure
    particle_t * d_particles;
    cudaMalloc((void **) &d_particles, n * sizeof(particle_t));

    set_size( n );

    init_particles( n, particles );

    double gridSize = sqrt(density * n);
    int binNumPerEdge = ceil(gridSize / cutoff);
    int binNumber = binNumPerEdge * binNumPerEdge;

    //Here we get rid of the bin struct. Instead, we use a particle_t array to express a bin. Because it is good for memory allocation.
    particle_t** d_bins;
    cudaMalloc((void ***) &d_bins, binNumber * binParticleMax * sizeof(particle_t *));
    int* binParticleNum;
    cudaMalloc((void **) &binParticleNum, binNumber * sizeof(int *));

    /****************Copy particles to GPU*************/
    cudaThreadSynchronize();
    double copy_time = read_timer( );

    // Copy the particles to the GPU
    cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);

    cudaThreadSynchronize();
    copy_time = read_timer( ) - copy_time;
    /****************Finish to copy particles to GPU*************/

    //
    //  simulate a number of time steps
    //
    cudaThreadSynchronize();
    double simulation_time = read_timer( );

    int blksParticle = (n + NUM_THREADS - 1) / NUM_THREADS; //Useful??? Will see.
    int blksBin = (binNumber + NUM_THREADS - 1) / NUM_THREADS;

    for( int step = 0; step < NSTEPS; step++ )
    {
        //
        //  reset the bins
        //
        cudaMemset(binParticleNum, 0, binNumber * sizeof(int));

        //
        // distribute particles into bins
        //
        distribute_particles_gpu <<< blksParticle, NUM_THREADS >>> (d_particles, d_bins, binNumPerEdge, binParticleNum, n);
        //
        //  compute forces
        //
        //compute_forces_bin_gpu <<< bin_blks, NUM_THREADS >>> (d_bins, num_particles_in_bins, n_bins_per_side, n_max_particles_per_bin);
        apply_force_per_bin_gpu <<< blksBin, NUM_THREADS >>> (d_bins, binNumPerEdge, binParticleNum)
        //
        //  move particles
        //
        move_gpu <<< blks, NUM_THREADS >>> (d_particles, n, gridSize);
        
        //
        //  save if necessary
        //
        if( fsave && (step%SAVEFREQ) == 0 ) {
            // Copy the particles back to the CPU
            cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
            save( fsave, n, particles);
        }
    }
    cudaThreadSynchronize();
    simulation_time = read_timer( ) - simulation_time;
    
    printf( "CPU-GPU copy time = %g seconds\n", copy_time);
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );
    
    if( fsum) 
        fprintf(fsum,"%d %g\n",n,simulation_time);
    
    free( particles );
    cudaFree(d_particles);
    cudaFree(d_bins);
    cudaFree(binParticleNum);
    if( fsave )
        fclose( fsave );
    
    return 0;
}