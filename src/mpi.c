#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <stdlib.h>

#define N 4
#define MAX_STRING 250

int main() {

    char greeting[MAX_STRING];
    int comm_sz;
    int my_rank;
    int n = N;
    printf("%d", n);

    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    MPI_Status status;

    int sum = 0;
    int* b;
 
    int* vect_1;
    int* vect_2;


    if (my_rank == 0) {

        //scanf("%d", &n);

        vect_1 = (int*)calloc(n, sizeof(int));
        vect_2 = (int*)calloc(n, sizeof(int));
        for (int i = 0; i < n; ++i)
            scanf("%d", &vect_1[i]);
        for (int i = 0; i < n; ++i)
            scanf("%d", &vect_2[i]);


        printf("\nGreetings from process %d of %d\n", my_rank, comm_sz);
        // for (int i = 1; i < comm_sz; ++i){
        //     MPI_Recv(greeting, MAX_STRING, MPI_CHAR, i, 0, MPI_COMM_WORLD, &status);
        //     printf("%s\n", greeting);
        // }

    } else {
        sprintf(greeting, "Greetings from process %d of %d!", my_rank, comm_sz);
        // MPI_Send(greeting, strlen(greeting) + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    b = (int*)calloc(n, sizeof(int));
    for(int j = 0; j < n; ++j) {
        b[j] = my_rank + j*2;
        sum += b[j];
    }
    MPI_Gather(b, n, MPI_INT, vect_1, n, MPI_INT, 0, MPI_COMM_WORLD);
    if (my_rank == 0)
        for (int i = 0; i < 2; ++i) {
            sprintf("%d!   ", i);
            for (int j = 0; j < n; ++j) {
                if (i == 0) {
                    sprintf("%d", vect_1[j]);
                } else sprintf("%d", vect_2[j]);
            }  
            sprintf("\n");  
        }
    MPI_Gather(&sum, 1, MPI_INT, vect_2, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (my_rank == 0) {
        printf(" S");
        for (int i = 0; i < n; ++i)
            printf("%d", vect_2[i]);
        printf('/n');
    }

    MPI_Finalize();

    free(b);
    free(vect_1);
    free(vect_2);

    return 0;
}
