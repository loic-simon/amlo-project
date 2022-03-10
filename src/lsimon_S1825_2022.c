#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define DOUBLE_PER_M256D sizeof(__m256d)/sizeof(double)         // 4

// Thread arguments structure
struct thread_args {
    double* U;
    int n;
    int mode;
};

// Mutex and associated global variable
pthread_mutex_t lock;
double glob_s;


// Linear computation
double rnorm(double *U, int n)
{
    double s = 0;
    for (int i = 0; i < n; i++)  s += sqrt(U[i]);
    return s;
}


// Vectorial computation with AVX
double vect_rnorm(double *U, int n)
{
    double s = 0;
    __m256d vect_U;

    for (int nn = 0; nn < n; nn += DOUBLE_PER_M256D) {
        // AVX256 = compute doubles 4 by 4

        if (nn > n - DOUBLE_PER_M256D) {
            // Less than 4 doubles remaining: need to fill with a mask
            __m256i mask = _mm256_setr_epi64x((n - nn) > 0 ? -1 : 1,
                                              (n - nn) > 1 ? -1 : 1,
                                              (n - nn) > 2 ? -1 : 1,
                                              (n - nn) > 3 ? -1 : 1);
            vect_U = _mm256_maskload_pd(U + nn, mask);
        } else {
            // 4+ doubles remaining: directly load
            vect_U = _mm256_load_pd(U + nn);
        }

        // Compute sqrt
        vect_U = _mm256_sqrt_pd(vect_U);

        // Sum
        for (int i = 0; i < DOUBLE_PER_M256D; i++)  s += vect_U[i];
    }

    return s;
}


// Function called in threads
void* callback(void* raw_args)
{
    struct thread_args *args = raw_args;

    // Compute in required mode
    double this_s = (args -> mode) ? vect_rnorm(args -> U, args -> n)
                                   : rnorm(args -> U, args -> n);

    // In mutex, add partial sum to global sum
    pthread_mutex_lock(&lock);
    glob_s += this_s;
    pthread_mutex_unlock(&lock);

    // Thread finished, exit
    pthread_exit(NULL);
    return NULL;
}


// Parallel computation (start and join threads)
double rnormPar(double *U, int n, int nb_threads, int mode)
{
    double s = 0;
    int remaining = n;
    int started_threads = 0;
    int rc;

    // Allocate threads and threads args
    pthread_t *threads = (pthread_t *)malloc(sizeof(pthread_t)*nb_threads);
    if (threads == NULL)  perror("Failed to malloc threads");
    struct thread_args *args = (struct thread_args *)malloc(sizeof(struct thread_args)*nb_threads);
    if (args == NULL)  perror("Failed to malloc args");

    // Compute threads repartition
    int n_per_thread = n / nb_threads + 1;
    while (n_per_thread % 4) n_per_thread++;
    // we need to cut U by bocks of 4, because of AVX if mode=1 that
    // only want aligned address blocks of length 4*sizeof(double) = 32UL

    // Init mutex
    rc = pthread_mutex_init(&lock, NULL);
    if (rc)  perror("Failed to init mutex");
    glob_s = 0;

    // Create threads
    for (int n_th = 0; n_th < nb_threads; n_th++) {
        if (remaining <= 0) break;      // Everything already consumed
        // Create args
        args[n_th].U = U + n_th*n_per_thread;
        args[n_th].n = (remaining > n_per_thread) ? n_per_thread : remaining;
        args[n_th].mode = mode;
        remaining -= n_per_thread;
        started_threads++;
        // Create thread
        rc = pthread_create(&(threads[n_th]), NULL, callback, (void *)&(args[n_th]));
        if (rc)  perror("Failed to start thread");
    }

    // Join threads (wait for all to complete)
    for (int n_th = 0; n_th < started_threads; n_th++) {
        rc = pthread_join(threads[n_th], NULL);
        if (rc)  perror("Failed to join thread");
    }
    s = glob_s;

    // Clean up
    pthread_mutex_destroy(&lock);
    free(threads);
    free(args);

    return s;
}


int main(int argc, char const *argv[])
{
    // Define parameters
    int n = 1024*1024;
    int nb_threads = 8;

    // Override parameters if args provided
    if (argc > 1)  sscanf(argv[1], "%d", &nb_threads);
    if (argc > 2)  sscanf(argv[2], "%d", &n);

    printf("nb_threads = %d\n", nb_threads);
    printf("n = %d\n", n);

    // Alloc an aligned memory to U, with blocks of 32 bits
    double* U = (double*)aligned_alloc(32, n * sizeof(double));

    // Get a "random" seed and compute U
    srand(time(0));
    for (int i = 0; i < n; i++) U[i] = (double)rand()/(double)(RAND_MAX);

    // Run all
    clock_t begin, end;
    double S, V, PS, PV;
    double time_S, time_V, time_PS, time_PV;

    // Sequencial
    begin = clock();
    S = rnorm(U, n);
    end = clock();
    time_S = (double)(end - begin) / CLOCKS_PER_SEC;

    // Vectorial
    begin = clock();
    V = vect_rnorm(U, n);
    end = clock();
    time_V = (double)(end - begin) / CLOCKS_PER_SEC;

    // Parallel
    begin = clock();
    PS = rnormPar(U, n, nb_threads, 0);
    end = clock();
    time_PS = (double)(end - begin) / CLOCKS_PER_SEC;

    // Vectorial + parallel
    begin = clock();
    PV = rnormPar(U, n, nb_threads, 1);
    end = clock();
    time_PV = (double)(end - begin) / CLOCKS_PER_SEC;

    // Print results
    printf("VALEURS\n");
    printf("Séquentiel (scalaire : %f  vectoriel : %f)  Parallèle (nb_thread : %d scalaire : %f  vectoriel : %f)\n", S, V, nb_threads, PS, PV);
    printf("TEMPS D’EXÉCUTION  \n");
    printf("Séquentiel (scalaire : %e  vectoriel : %e)  Parallèle (nb_thread : %d scalaire : %e  vectoriel : %e)\n", time_S, time_V, nb_threads, time_PS, time_PV);
    printf("Accélération (vectoriel : %f  multithread : %f vectoriel + multithread : %f)\n", time_S/time_V, time_S/time_PS, time_S/time_PV);

    // Clean up
    free(U);

    return 0;
}
