/*
    Problem: 2D parallel n-body simulation using the Leap frog method
    Assumption: Number of particles (bodies) is evenly divisible by number of processes. 
    Program exits if number of bodies is not evenly divisible by number of bodies
    Author: Halidu Abdulai
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

const int ROOT = 0;        // Process 0 is root process

const double G = 6.67259e-7;  /* Gravitational constant (should be e-10 but modified to get more action */
const double dt = 1.0;         /* Length of timestep */

/* Writes out positions (x,y) of N particles to the file fn
   Returns zero if the file couldn't be opened, otherwise 1 */
int write_particles(int N, double* X, double* Y, char* fn) {
    FILE* fp;
    /* Open the file */
    if ((fp = fopen(fn, "w")) == NULL) {
        printf("Couldn't open file %s\n", fn);
        return 0;
    }
    /* Write the positions to the file fn */
    for (int i = 0; i < N; i++) {
        fprintf(fp, "%3.2f %3.2f \n", X[i], Y[i]);
    }
    fprintf(fp, "\n");
    fclose(fp);  /* Close the file */
    return(1);
}

// Calculates the distance between two bodies
double dist(double px, double py, double qx, double qy) {
    return sqrt(pow(px - qx, 2) + pow(py - qy, 2));
}

/* Computes forces between bodies */
void ComputeForce(int start_idx, int end_idx, int N, double* X, double* Y, double* mass, double* Fx, double* Fy) {
    const double mindist = 0.0001;  /* Minimal distance of two bodies of being in interaction*/

    for (int i = start_idx; i < end_idx; i++) {      // Compute force for current process' own bodies
        Fx[i] = Fy[i] = 0.0;             // Initialize force vector to zero
        for (int j = 0; j < N; j++) {   // The force on a body i is the sum of forces from all other bodies j
            if (i != j) {                  //     but not from it self
                // Distance between points i and j
                double r = dist(X[i], Y[i], X[j], Y[j]);

                if (r > mindist) {        // Very near-distance forces are ignored
                    double r3 = pow(r, 3);     
                    Fx[i] += G * mass[i] * mass[j] * (X[j] - X[i]) / r3;  // Add the force experienced by body i due to body j on the x-axis
                    Fy[i] += G * mass[i] * mass[j] * (Y[j] - Y[i]) / r3;  // Add the force experienced by body i due to body j on the y-axis
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    int my_rank, comm_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
   
     int N, timesteps;
    // Get number of bodies and timesteps from the command line if user passed them. (This is just to make the program flexible)
    if (argc >= 3)
    {
        N = atoi(argv[1]);
        timesteps = atoi(argv[2]);
    }
    else {        // default is 1000 particles and 1000 timesteps 
        N = 1000;
        timesteps = 1000;
    }
    // Program won't  proceed if number of bodies is not evenly divisible by number of processes
    if ((N % comm_size) != 0) {
        if (my_rank == ROOT) {
            // Only master process reports the error
            fprintf(stderr, "Number of bodies must be evenly divisible by number of processes!\n");
        }
        exit(0);       // all processes exit 
    }

    // Initial distances for all bodies is between 0-100 
    const double size = 100.0;

    double* mass, * X, * Y, * Vx, * Vy, * Fx, * Fy ;

    // number of bodies must be evenly divisible by number of processes
    int bodies_per_process = N / comm_size;
    int start_idx = my_rank * bodies_per_process;
    int end_idx = (my_rank == comm_size - 1) ? N : start_idx + bodies_per_process;


    /* Allocate memory for variables  */
    mass = (double*)calloc(N, sizeof(double));  // Mass 
    X = (double*)calloc(N, sizeof(double));  // Position (x,y) at current time step
    Y = (double*)calloc(N, sizeof(double));
    Vx = (double*)calloc(N, sizeof(double));  // Velocities at current time step 
    Vy = (double*)calloc(N, sizeof(double));
    Fx = (double*)calloc(N, sizeof(double));  // Forces at current time step
    Fy = (double*)calloc(N, sizeof(double));

    // Process 0 performs initialization.
    if (my_rank == 0) {
        /* Initialize mass and position of bodies */
        unsigned short int seedval[3] = { 7, 7, 7 };
        seed48(seedval);
        for (int i = 0; i < N; i++) {
            mass[i] = 1000.0 * drand48();   // 0 <= mass < 1000
            X[i] = size * drand48();      // 0 <= X < 100
            Y[i] = size * drand48();      // 0 <= Y < 100 

        }
        // Write intial particle coordinates to a file
        write_particles(N, X, Y, "parallel_initial_pos.txt");
    }
    // The ROOT process broadcasts the masses and initial postions (X, Y) of all bodies to all the processes
    MPI_Bcast(mass, N, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);       // ROOT is defined above as 0 (process 0)
    MPI_Bcast(X, N, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(Y, N, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    
    double start = MPI_Wtime();    // Start timer
    // Compute the initial forces that we get - each process computes for its own bodies
    ComputeForce(start_idx, end_idx, N, X, Y, mass, Fx, Fy);

    // Set up the velocity vectors caused by initial forces for Leapfrog method - each process computes for its own bodies
    for (int i = start_idx; i < end_idx; i++) {
        Vx[i] = Vy[i] = 0.0;
        Vx[i] = 0.5 * dt * Fx[i] / mass[i];
        Vy[i] = 0.5 * dt * Fy[i] / mass[i];
    }

    // iterate for number of timesteps - computing and updating the forces, velocites and positions of the bodies
    for (int t = 0; t < timesteps; t++) {
        // Calculate new positions - each process computes for own bodies
        for (int i = start_idx; i < end_idx; i++) {
            X[i] = X[i] + Vx[i] * dt;
            Y[i] = Y[i] + Vy[i] * dt;
        }

        // Gather distances globally - each process only sends the part it is in charge of updating
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, X, bodies_per_process, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, Y, bodies_per_process, MPI_DOUBLE, MPI_COMM_WORLD);

        //Compute forces
        ComputeForce(start_idx, end_idx, N, X, Y, mass, Fx, Fy);

        // Update velocities 
        for (int i = start_idx; i < end_idx; i++) {
            Vx[i] = Vx[i] + Fx[i] * dt / mass[i];
            Vy[i] = Vy[i] + Fy[i] * dt / mass[i];

        }
    }
    double end = MPI_Wtime();     // stop timer
    if (my_rank == 0) {
        write_particles(N, X, Y, "parallel_final_pos.txt");    // process 0 writes the final positions of the bodies to file
        printf("Simulation Time: %6.2f seconds\n", (end - start));   // print how long the process took
    }
    // free memory
    free(mass);
    free(Fx);
    free(Fy);
    free(Vx);
    free(Vy);
    free(X);
    free(Y);
    // Finalize MPI
    MPI_Finalize();
    return 0;
}