#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

int main(int argc, char *argv[]) {
    srand(time(NULL));  // Seed random number generator
    
    
    int winner = 0;  // Initialize winner
    #pragma omp parallel  reduction(max:winner) num_threads(4)
    {
        int thread_winner = (rand() % 1000) + omp_get_thread_num();
        printf("Thread %d has chosen %d\n", omp_get_thread_num(), thread_winner);

        winner = thread_winner;  // Apply reduction
    }

    printf("Final winner: %d\n", winner);
    return 0;
}
