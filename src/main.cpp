#include <iostream>
#include <fstream>
#include <pthread.h>
#include <semaphore.h>
#include <cmath>
#include <algorithm>

#include <iostream>
#include <chrono>


long** read_matrix(char* path, long& n, long& m);
void clean_matrix(long** matrix, const long n);
const void print_matrix(long** matrix, const long n, const long m);
void null_matrix(long** matrix, const long n, const long m);
void calculate_boarders(long n_1, long m_1, long m_2);
void clean_boarders();
 
long number_of_threads = 1;
long* boarders_row_1;
long* boarders_column_1;
long* boarders_column_2;
pthread_mutex_t mutex;
sem_t** semaphores;
pthread_mutex_t** mutexes;

struct matrix
{
    long** matrix;
    long n;
    long m;
};

struct matrix_double
{
    double** matrix;
    long n;
    long m;
};

struct matrix_args
{
    matrix* matrix_1;
    matrix* matrix_2;
    long** result_matrix;
} Matrix_pair;

struct position
{
    long x;
    long y;
};

position* get_position(long current_rank, long block_size_m, long block_size_n);

void sum_matrix_rows(long** matrix_1,     long** matrix_2,
                     long row_1_start,    long column_1_start,
                     long column_2_start, long row_1_end,
                     long column_1_end,   long column_2_end,
                     long** result_matrix);
                          
void sum_matrix_columns(long** matrix_1,     long** matrix_2,
                        long column_1_start, long row_1_start,
                        long column_2_start, long column_1_end,
                        long row_1_end,      long column_2_end,
                        long** result_matrix);

void sum_matrix_blocks(long** matrix_1, long** matrix_2,
                       long block_row_start,
                       long block_column_start,
                       long block_common_index,
                       long block_size,
                       long** result_matrix);

void* Routine_blocks(void* rank);

void* Routine_rows(void* rank);

void* Routine_columns(void* rank);

void QR(matrix* matrix);
 

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "No requered params! Program go to rest." << std::endl;
        return -1;
    }

    long n_1, m_1, n_2, m_2;
    n_1 = m_1 = n_2 = m_2 = 0;
    long** matrix_1 = nullptr;
    long** matrix_2 = nullptr;
    try {
        matrix_1 = read_matrix(argv[1], n_1, m_1);
        matrix_2 = read_matrix(argv[2], n_2, m_2);
    } catch(const std::exception& e) {
            std::cout << "Ooops exception!";
            return -1;
    }

    if (m_1 != n_2) {
        std::cout << "Matrix sizes not valid! Work was stopped." << std::endl;
        clean_matrix(matrix_1, n_1);
        clean_matrix(matrix_2, n_2);
        return -1;
    }


    matrix* matrix_struct_1 = (matrix*)malloc(sizeof(matrix));
    matrix* matrix_struct_2 = (matrix*)malloc(sizeof(matrix));

    matrix_struct_1->matrix = matrix_1;
    matrix_struct_1->n = n_1;
    matrix_struct_1->m = m_1;

    matrix_struct_2->matrix = matrix_2;
    matrix_struct_2->n = n_2;
    matrix_struct_2->m = m_2;

    long** result_matrix = nullptr;
    result_matrix = (long**)calloc(n_1, sizeof(long*));
    for (long i = 0; i < n_1; ++i) {
        result_matrix[i] = (long*)calloc(m_2, sizeof(long));
    }

    long** result_matrix_2 = nullptr;
    result_matrix_2 = (long**)calloc(n_1, sizeof(long*));
    for (long i = 0; i < n_1; ++i) {
        result_matrix_2[i] = (long*)calloc(m_2, sizeof(long));
    }

    long** result_matrix_3 = nullptr;
    result_matrix_3 = (long**)calloc(n_1, sizeof(long*));
    for (long i = 0; i < n_1; ++i) {
        result_matrix_3[i] = (long*)calloc(m_2, sizeof(long));
    }

    Matrix_pair.matrix_1 = matrix_struct_1;
    Matrix_pair.matrix_2 = matrix_struct_2;
    Matrix_pair.result_matrix = result_matrix;


    if (argc >= 4)
        number_of_threads = strtol(argv[3], NULL, 10);

    std::chrono::time_point<std::chrono::steady_clock> begin, end;

    //print_matrix(matrix_1, n_1, m_1);
    //print_matrix(matrix_2, n_2, m_2);

    pthread_t* thread_handles = (pthread_t*)malloc(number_of_threads * sizeof(pthread_t));
    semaphores = (sem_t**)malloc(n_1 * sizeof(sem_t*));
    mutexes = (pthread_mutex_t**)malloc(n_1 * sizeof(pthread_mutex_t*));
    for (int i = 0; i < n_1; ++i) {
        semaphores[i] = (sem_t*)malloc(m_2 * sizeof(sem_t));
        mutexes[i] = (pthread_mutex_t*)malloc(m_2 * sizeof(pthread_mutex_t));
        for (int j = 0; j < m_2; ++j) {
            sem_init(&semaphores[i][j], 0, 0);
            pthread_mutex_init(&mutexes[i][j], NULL);
        }  
    }  

    begin = std::chrono::steady_clock::now();
    for(long thread = 0; thread < number_of_threads; ++thread)
        pthread_create(&thread_handles[thread], NULL, Routine_rows, (void*)thread);                          

    for(long thread = 0; thread < number_of_threads; ++thread)
    {
        pthread_join(thread_handles[thread], NULL);
    }
    end = std::chrono::steady_clock::now();
    std::cout << "The time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms\n";

    //print_matrix(result_matrix, n_1, m_2);
    //print_matrix(Matrix_pair.result_matrix, n_1, m_2);
    //clean_matrix(result_matrix, m_1);
    //null_matrix(result_matrix, n_1, m_2);

    Matrix_pair.result_matrix = result_matrix_2;

    pthread_mutex_init(&mutex, NULL);
    begin = std::chrono::steady_clock::now();
    if (number_of_threads <= n_1 * m_2) {
        for(long thread = 0; thread < number_of_threads; ++thread)
            pthread_create(&thread_handles[thread], NULL, Routine_blocks, (void*)thread);                          

        for(long thread = 0; thread < number_of_threads; ++thread)
        {
            pthread_join(thread_handles[thread], NULL);
        }
    }
    else {
        std::cout << "Oooops problem whith matrix size for blocks methods!" << std::endl;
    }
    end = std::chrono::steady_clock::now();
    std::cout << "The time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms\n";
    //clean_boarders();

    //print_matrix(result_matrix, n_1, m_2);
    //clean_matrix(result_matrix, m_1);
    //null_matrix(result_matrix, n_1, m_2);

    Matrix_pair.result_matrix = result_matrix_3;
    begin = std::chrono::steady_clock::now();
    for(long thread = 0; thread < number_of_threads; ++thread)
        pthread_create(&thread_handles[thread], NULL, Routine_columns, (void*)thread);                          

    for(long thread = 0; thread < number_of_threads; ++thread)
    {
        pthread_join(thread_handles[thread], NULL);
    }
    end = std::chrono::steady_clock::now();
    std::cout << "The time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms\n";


    //print_matrix(result_matrix, n_1, m_2);
    //print_matrix(result_matrix_2, n_1, m_2);
    //print_matrix(result_matrix_3, n_1, m_2);
    pthread_mutex_destroy(&mutex);

    for (int i = 0; i < n_1; ++i) {
        for (int j = 0; j < m_2; ++j) {
            sem_destroy(&semaphores[i][j]);
            pthread_mutex_destroy(&mutexes[i][j]);
        }  
    }

    long error_count = 0;
    for (long i = 0; i < n_1; ++i) {
        for (long j = 0; j < m_2; ++j) {
            if (result_matrix[i][j] != result_matrix_2[i][j]) {
                std::cout << result_matrix[i][j] << " vs " << result_matrix_2[i][j] << std::endl;
                ++error_count;
            }
        }
    }
    std::cout << "Errors 1 vs 2: " << error_count << ", From: " << n_1*m_2 << std::endl;

    error_count = 0;
    for (long i = 0; i < n_1; ++i) {
        for (long j = 0; j < m_2; ++j) {
            if (result_matrix[i][j] != result_matrix_3[i][j]) {
                ++error_count;
            }
        }
    }
    std::cout << "Errors 1 vs 3: " << error_count << ", From: " << n_1*m_2 << std::endl;


    
    clean_matrix(matrix_1, n_1);
    clean_matrix(matrix_2, n_2);
    clean_matrix(result_matrix, n_1);
    clean_matrix(result_matrix_2, n_1);
    clean_matrix(result_matrix_3, n_1);

    free(matrix_struct_1);
    free(matrix_struct_2);
    free(thread_handles);

    return 0;
}

long** read_matrix(char* path, long& n, long& m) {
    long** matrix = nullptr;
    std::ifstream in(path);
    if (in.is_open())
    {
        try {
            in >> n >> m;
        } catch (const std::exception& e) {
            std::cerr << "N and M parsing error: " << e.what() << std::endl;
            throw e;
        }
        matrix = (long**)malloc(sizeof(long*) * n);
        for (long i = 0; i < n; ++i) {
            matrix[i] = (long*)calloc(m, sizeof(long));
        }
        try
        {
            for (long i = 0; i < n; i++)
                for (long j = 0; j < m; ++j)
                    in >> matrix[i][j];
        }
        catch(const std::exception& e)
        {
            std::cerr << "Matrix element parsing error: " << e.what() << '\n';
            throw e;
        }
    }
    in.close();
    return matrix;
}

void clean_matrix(long** matrix, const long n) {
    for (long i = 0; i < n; ++i)
        free(matrix[i]);
    free(matrix);   
}

const void print_matrix(long** matrix, const long n, const long m) {
    std::cout << "==================================================" << std::endl;
    for (long i = 0; i < n; ++i) {
        for (long j = 0; j < m; ++j)
            std::cout << matrix[i][j] << '\t';
        std::cout << std::endl;
    }
    std::cout << "==================================================" << std::endl;
        
}

void null_matrix(long** matrix, const long n, const long m) {
    for (long i = 0; i < n; ++i) {
        for (long j = 0; j < m; ++j)
            matrix[i][j] = 0;
    }   
}

void sum_matrix_rows(long** matrix_1, long** matrix_2,
                     long row_1_start, long column_1_start,
                     long column_2_start, long row_1_end,
                     long column_1_end, long column_2_end,
                     long** result_matrix) {

    for(long row_1 = row_1_start; row_1 < row_1_end && row_1 < Matrix_pair.matrix_1->n; ++row_1) 
        for(long column_2 = column_2_start; column_2 < column_2_end && column_2 < Matrix_pair.matrix_2->m; ++column_2) 
            for(long column_1 = column_1_start; column_1 < column_1_end && column_1 < Matrix_pair.matrix_1->m; ++column_1) {
                result_matrix[row_1][column_2] += matrix_1[row_1][column_1] * matrix_2[column_1][column_2];
            }
}

void sum_matrix_columns(long** matrix_1, long** matrix_2,
                     long column_1_start, long row_1_start,
                     long column_2_start, long column_1_end,
                     long row_1_end, long column_2_end,
                     long** result_matrix) {
    
    for(long column_1 = column_1_start; column_1 < column_1_end && column_1 < Matrix_pair.matrix_1->m; ++column_1) {
        for(long row_1 = row_1_start; row_1 < row_1_end && row_1 < Matrix_pair.matrix_1->n; ++row_1) {
             pthread_mutex_lock(&mutexes[row_1][0]);
             for(long column_2 = column_2_start; column_2 < column_2_end && column_2 < Matrix_pair.matrix_2->m; ++column_2) {
                //sem_post(&semaphores[row_1][column_2]);
                //pthread_mutex_lock(&mutex);
                //pthread_mutex_lock(&mutexes[row_1][column_2]);
                //pthread_mutex_lock(&mutexes[row_1][0]);
                result_matrix[row_1][column_2] += matrix_1[row_1][column_1] * matrix_2[column_1][column_2];
                //pthread_mutex_unlock(&mutexes[row_1][0]);
                //pthread_mutex_unlock(&mutexes[row_1][column_2]);
                //pthread_mutex_unlock(&mutex);
                //sem_wait(&semaphores[row_1][column_2]);
             }
             pthread_mutex_unlock(&mutexes[row_1][0]);
        }
    }
}

void sum_matrix_blocks(long** matrix_1, long** matrix_2,
            long block_row_start,
            long block_column_start,
            long block_common_index,
            long block_size_n,
            long block_size_m,
            long** result_matrix) {

    for(long i = 0; i < block_size_n && block_row_start + i < Matrix_pair.matrix_1->n; ++i) {
        for(long j = 0; j < block_size_m && block_column_start + j < Matrix_pair.matrix_2->m; ++j) {
            pthread_mutex_lock(&mutexes[i][j]);
            for(long k = 0; k < block_size_n && k < block_size_m && block_common_index + k < Matrix_pair.matrix_1->n && block_common_index + k < Matrix_pair.matrix_2->m; ++k) {
                result_matrix[block_row_start+i][block_column_start+j] +=
                  matrix_1[block_row_start+i][block_common_index+k] *
                  matrix_2[block_common_index+k][block_column_start+j];
            }
            pthread_mutex_unlock(&mutexes[i][j]);
        }
    }
}

void* Routine_blocks(void* rank) {
    long current_rank = (long)rank;
    long max_rank = number_of_threads-1;
    long block_size = long(sqrt(double(number_of_threads)));
    long block_size_n = Matrix_pair.matrix_1->n / block_size;
    long block_size_m = Matrix_pair.matrix_1->m / block_size;

    if (!block_size_n) block_size_n = 1;
    if (!block_size_m) block_size_m = 1;

    long block_row_start = 0;
    long block_column_start = 0;
    long block_common_index = 0;

    // block_row_start = (current_rank / number_of_threads) * block_size_n;
    // block_column_start = (current_rank % number_of_threads)* block_size_m;

    position* current_position = get_position(current_rank, block_size_m, block_size_n);
    block_row_start = current_position->x;
    block_column_start = current_position->y;
    free(current_position);

    for (block_common_index; block_common_index < Matrix_pair.matrix_1->n && block_common_index < Matrix_pair.matrix_2->m; block_common_index += std::min(block_size_n, block_size_m)) {
        sum_matrix_blocks(Matrix_pair.matrix_1->matrix,
                          Matrix_pair.matrix_2->matrix,
                          block_row_start,
                          block_column_start,
                          block_common_index,
                          block_size_n,
                          block_size_m,
                          Matrix_pair.result_matrix);
    }



    current_position = get_position(number_of_threads-1, block_size_m, block_size_n);
    block_row_start = current_position->x;
    block_column_start = current_position->y;

    free(current_position);

    long rows = Matrix_pair.matrix_1->n - (block_row_start + block_size_n + 1);
    int number_of_rows = rows / number_of_threads;
    long first_row = (block_row_start + block_size_n) + current_rank * number_of_rows;
    long last_row = (block_row_start + block_size_n) + (current_rank + 1) * number_of_rows;

    if (current_rank == (number_of_threads-1) && last_row < Matrix_pair.matrix_1->n)
        last_row = Matrix_pair.matrix_1->n;

    sum_matrix_rows(Matrix_pair.matrix_1->matrix, Matrix_pair.matrix_2->matrix, first_row, 0, 0, last_row,
                    Matrix_pair.matrix_1->m, Matrix_pair.matrix_2->m, Matrix_pair.result_matrix);

    // rows = block_row_start + block_size_n + 1;
    // number_of_rows = rows / number_of_threads;
    // long save_first_row = first_row;
    // first_row = current_rank * number_of_rows;
    // last_row = (current_rank + 1) * number_of_rows;

    long column_index = Matrix_pair.matrix_1->m - (Matrix_pair.matrix_1->m % block_size_m);
    long columns = Matrix_pair.matrix_1->m - (column_index + 1);

    int number_of_columns = columns / number_of_threads;
    long first_column = column_index + current_rank * number_of_columns;
    long last_column = column_index + (current_rank + 1) * number_of_columns;

    if (current_rank == (number_of_threads-1) && last_column < Matrix_pair.matrix_1->m)
        last_column = Matrix_pair.matrix_1->m;

    sum_matrix_columns(Matrix_pair.matrix_1->matrix, Matrix_pair.matrix_2->matrix, 0, 0, first_column, Matrix_pair.matrix_1->m, first_row, last_column, Matrix_pair.result_matrix);

    rows = block_size_n; 
    number_of_rows = rows / number_of_threads;
    if (number_of_rows == 0)
        number_of_rows = 1;

    long save_first_row = first_row;
    first_row = block_row_start + current_rank * number_of_rows;
    last_row = block_row_start + (current_rank + 1) * number_of_rows;

    if (current_rank == max_rank && last_row < save_first_row)
        last_row = save_first_row;

    if (first_row < save_first_row)
        sum_matrix_rows(Matrix_pair.matrix_1->matrix, Matrix_pair.matrix_2->matrix, first_row, 0, block_column_start + block_size_m, last_row, Matrix_pair.matrix_1->m, column_index, Matrix_pair.result_matrix);

    // long last_column_index = block_column_start + block_size_m - 1;
    // columns = Matrix_pair.matrix_1->m - (last_column_index + 1);
    // number_of_columns = columns / number_of_threads;
    // if (number_of_columns == 0)
    //     number_of_columns = 1;

    // first_column = last_column_index + current_rank * number_of_columns;
    // last_column = last_column_index + (current_rank + 1) * number_of_columns;

    // if (last_column > column_index)
    //     last_column = column_index;    

    // if (first_column < column_index)

    //     sum_matrix_columns(Matrix_pair.matrix_1->matrix, Matrix_pair.matrix_2->matrix, first_column, block_row_start, last_column, Matrix_pair.matrix_1->m, first_row, column_index, Matrix_pair.result_matrix);
    // if (current_rank == (number_of_threads-1) && last_row < save_first_row)
    //     last_row = save_first_row;

    // sum_matrix_rows(Matrix_pair.matrix_1->matrix, Matrix_pair.matrix_2->matrix, first_row, first_column, 0, last_row,
    //                 Matrix_pair.matrix_1->m, Matrix_pair.matrix_2->m, Matrix_pair.result_matrix);
    // for (long i = current_rank; i < current_rank+1; ++i) {
    //     for (long j = current_rank; j < current_rank+1; ++j) {
    //         for (long c = current_rank; c < current_rank+1; ++c) {
    //             pthread_mutex_lock(&mutex);
    //             sum_matrix_rows(Matrix_pair.matrix_1->matrix, Matrix_pair.matrix_2->matrix, 
    //                             boarders_row_1[i], boarders_column_1[j], boarders_column_2[c],
    //                             boarders_row_1[i+1], boarders_column_1[j+1], boarders_column_2[c+1],
    //                             Matrix_pair.result_matrix);
    //             std::cout << boarders_row_1[i] << ' ' << boarders_column_1[j] << ' ' << boarders_column_2[c] << " ends: " <<
    //                          boarders_row_1[i+1] << ' ' << boarders_column_1[j+1] << ' ' << boarders_column_2[c+1] << " runk: " << current_rank << std::endl;
    //             pthread_mutex_unlock(&mutex);
    //         }
    //     }
    // }
    // sum_matrix_rows(Matrix_pair.matrix_1->matrix, Matrix_pair.matrix_2->matrix, 
    //                 boarders_row_1[current_rank], boarders_column_1[current_rank], boarders_column_2[current_rank],
    //                 boarders_row_1[current_rank+1], boarders_column_1[current_rank+1], boarders_column_2[current_rank+1],
    //                 Matrix_pair.result_matrix); 
    return NULL;
}

void* Routine_rows(void* rank) {

    long rows = Matrix_pair.matrix_1->n;
    long current_rank = (long)rank;

    int number_of_rows = rows / number_of_threads;
    long first_row = current_rank * number_of_rows;
    long last_row = (current_rank + 1) * number_of_rows;

    if (current_rank == (number_of_threads-1))
        last_row = rows;

    sum_matrix_rows(Matrix_pair.matrix_1->matrix, Matrix_pair.matrix_2->matrix, first_row, 0, 0, last_row,
                    Matrix_pair.matrix_1->m, Matrix_pair.matrix_2->m, Matrix_pair.result_matrix);

    return NULL;
}


void* Routine_columns(void* rank) {

    long columns = Matrix_pair.matrix_1->m;
    long current_rank = (long)rank;

    int number_of_columns = columns / number_of_threads;
    long first_column = current_rank * number_of_columns;
    long last_column = (current_rank + 1) * number_of_columns;

    if (current_rank == (number_of_threads-1))
        last_column = columns;

    sum_matrix_columns(Matrix_pair.matrix_1->matrix, Matrix_pair.matrix_2->matrix, first_column, 0, 0, last_column,
                       Matrix_pair.matrix_1->n, Matrix_pair.matrix_2->m, Matrix_pair.result_matrix);

    return NULL;
}

void calculate_boarders(long n_1, long m_1, long m_2) {
    long blocks_num = number_of_threads;
    boarders_row_1 = (long*)calloc(blocks_num+1, sizeof(long));
    boarders_column_1 = (long*)calloc(blocks_num+1, sizeof(long));
    boarders_column_2 = (long*)calloc(blocks_num+1, sizeof(long));
    int i = 0;
    long board = 0;
    long block_row_1_size = n_1 / blocks_num;
    long block_column_1_size = m_1 / blocks_num;
    long block_column_2_size = m_2 / blocks_num;

    while (i != blocks_num && board < n_1) {
        boarders_row_1[i] = board;
        ++i;
        board += block_row_1_size;
        if (i == blocks_num) {
            boarders_row_1[blocks_num] = n_1;
            break;
        }
    }

    board = 0;
    i = 0;
    while (i != blocks_num && board < m_1) {
        boarders_column_1[i] = board;
        ++i;
        board += block_column_1_size;
        if (i == blocks_num) {
            boarders_column_1[blocks_num] = m_1;
            break;
        }
    }

    board = 0;
    i = 0;
    while (i != blocks_num && board < m_2) {
        boarders_column_2[i] = board;
        ++i;
        board += block_column_2_size;
        if (i == blocks_num) {
            boarders_column_2[blocks_num] = m_2;
            break;
        }
    }
}

void clean_boarders() {
    free(boarders_row_1);
    free(boarders_column_1);
    free(boarders_column_2);
}


position* get_position(long current_rank, long block_size_m, long block_size_n) {
    long block_column_start = 0;
    long block_row_start = 0;
    long block_i = 0; 
    while (current_rank > 0) {
        block_i += block_size_m;
        if (block_i < Matrix_pair.matrix_1->m && block_i + block_size_m <= Matrix_pair.matrix_1->m) {
            block_column_start += block_size_m;
            --current_rank;
        }
        else {            
            block_column_start = 0;
            block_i = 0;
            block_row_start += block_size_n;
            --current_rank;
        }
    }
    position* result = (position*)malloc(sizeof(position));
    result->x = block_row_start;
    result->y = block_column_start;
    return result;
}

double column_multiplication(matrix* vectors, long column_i, long column_j) {
    double answer = 0;
    for (long i = 0; i < vectors->n; i++) {
        answer += vectors->matrix[i][column_i] * vectors->matrix[i][column_j];
    }
    return answer;
}


double* column_proj(matrix*, long* column_i, long* column_j) {
    double cf = column_multiplication(matrix)
}

void QR(matrix* matrix) {
    

}

