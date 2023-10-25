#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <inttypes.h>
#include <omp.h> 

#define DEBUG 0            // Set to 1 if more info is desired
#define NR(i) (2*i+1)     // The number represented by position i 
#define POS(k) (k/2)      // The position in the array of number k

int main(int argc, char* argv[]) {

    int64_t i, t, k, N, N_pos;

    long int  nr_primes, lastprime = 0;
    char* prime = NULL;
    const char unmarked = (char)0;
    const char marked = (char)1;
    double start, stop;

    if (argc < 2) {
        printf("Usage: %s Number of Threads\n", argv[0]);
        exit(-1);
    }

    N = atoll(argv[1]);
    N_pos = (N - 3) / 2 + 2;   // Number of odd numbers between 2 and N - first index is used to store the number 2

    start = omp_get_wtime();   // start timer

    prime = (char*)malloc(N_pos * sizeof(char));       // allocate memory for odd numbers array
    // Check if allocation of memory succeeded
    if (prime == NULL) {
        printf("Could not allocate %jd chars of memory\n", N_pos);
        exit(-1);
    }

    prime[0] = marked;   // Mark index 0 since it is not used. 

    #pragma omp parallel for private(i) shared(prime) 
    for (i = 1; i < N_pos; i++) {   // initially, all numbers from index 1 to N_pos - 1 are set to unmarked
        prime[i] = unmarked;
    }

    int max_num = (int)sqrt((double)N);   // the largest odd number to check for is less than or equal to the sqrt of N
    // OpenMP needs to know the total number of iterations, in order to distribute the job.
    int total_iter = 0;
    for (total_iter = 0; NR(total_iter) < max_num; total_iter++);

    #pragma omp parallel for private(i,t, k) shared(prime) 
    // Position i in the array prime now corresponds to the number 2*i+1
    for (i = 1; i <= total_iter; i++) {
        if (prime[i] == unmarked) {
            if (DEBUG) printf("Marking multiples of %jd:", NR(i));
            t = NR(i);  // Position i corresponds to the number t
            for (k = POS(t * t); k < N_pos; k += t) {
                prime[k] = marked;   // Mark the multiples of i
                if (DEBUG) printf("%jd ", NR(k));
            }
            if (DEBUG) printf("\n");
        }
    }

    nr_primes = 1;   // Remember to count 2, as it is also a prime number. 
    // perform an addition reduction operation on nr_primes and a maximum reduction operation on lastprime
    #pragma omp parallel for private(i) reduction(+:nr_primes) reduction(max:lastprime)
    for (i = 1; i < N_pos; i++) {
        if (prime[i] == unmarked) {
            lastprime = NR(i);   // largest prime number so far is the unmarked number at postion i
            nr_primes++;   // increase number of primes by one if number at position i is not unmarked.
        }
    }

    stop = omp_get_wtime();   // stop timer
    printf("Time taken: %6.6f s\n", (stop - start));

    if (DEBUG) {
        printf("\nPrime numbers smaller than or equal to %jd are\n", N);
        printf("%d ", 2);
        for (i = 1; i < N_pos; i++) {
            if (prime[i] == unmarked) {
                printf("%jd ", NR(i));
            }
        }
    }
    // Prime information on the number of primes found and the largest prime among them
    printf("\nThere are %ld primes smaller than or equal to %jd\n", nr_primes, N);   // %jd is used to print a number of type "int64_t"
    printf("The largest of these primes is %ld \n", lastprime);

    free(prime); // Free the allocated memory

    return 0;
}
