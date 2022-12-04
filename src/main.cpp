#include <iostream>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iomanip>

#include <pthread.h>
#include <semaphore.h>


// Общие функции.
long** read_matrix(char* path, long& n, long& m);
void clean_matrix(long** matrix, const long n);
void clean_matrix(double** matrix, const long n);
const void print_matrix(long** matrix, const long n, const long m);
const void print_matrix(double** matrix, const long n, const long m);
void null_matrix(long** matrix, const long n, const long m);

// Количество потоков.
long number_of_threads = 1;

// Семафоры и мьютексы для сихронизации.
sem_t semaphore;
sem_t semaphore_2;
pthread_mutex_t** mutexes;

// Матрица целых чисел.
struct matrix
{
    long** matrix;
    long n;
    long m;
};

// Матрица с иррациональными числами.
struct matrix_double
{
    double** matrix;
    long n;
    long m;
};

// Общая структура данных из двух целочисленных матриц для многопоточности.
struct matrix_args
{
    matrix* matrix_1;
    matrix* matrix_2;
    long** result_matrix;
} Matrix_pair;

// Общая структура данных из двух double матриц для многопоточности.
struct matrixes_double 
{
    matrix_double* matrix_1;
    matrix_double* matrix_2;
} Matrix_for_QR, Matrixes_Q;

// Структура, представляюща собой коардинаты x, y на матрице.
struct position
{
    long x;
    long y;
};

// QR разложение матрицы.
struct QR
{
    double** Q;
    double** R;
} QR_matrix;

// Гибкое умножение матриц по строкам.
void sum_matrix_rows(long** matrix_1,     long** matrix_2,
                     long row_1_start,    long column_1_start,
                     long column_2_start, long row_1_end,
                     long column_1_end,   long column_2_end,
                     long** result_matrix);
                          
// Гибкое умножение матриц по столбцам.
void sum_matrix_columns(long** matrix_1,     long** matrix_2,
                        long column_1_start, long row_1_start,
                        long column_2_start, long column_1_end,
                        long row_1_end,      long column_2_end,
                        long** result_matrix);

// Функция для получения начала блока для текущего потока.
position* get_position(long current_rank, long block_size_m, long block_size_n);

// Гибкое умножение матриц по блокам.
void sum_matrix_blocks(long** matrix_1, long** matrix_2,
                       long block_row_start,
                       long block_column_start,
                       long block_common_index,
                       long block_size,
                       long** result_matrix);

// Распределение строк для каждого потока.
void* Routine_rows(void* rank);

// Распределение столбцов для каждого потока.
void* Routine_columns(void* rank);

// Распределение блоков для каждого потока и вызов функции расчёта.
void* Routine_blocks(void* rank);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// QR разложение                                                                                                         //
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double column_scalar_multiplication(matrix_double* vectors, long column_i, matrix_double* vectors_second, long column_j);//
double* column_proj(matrix_double* b, matrix_double* a, long column_i, long column_j);                                   //
void QR(matrix_double* vectors, pthread_t* pthread_handler);                                                             //
void vector_subtraction(matrix_double* vectors, long column_i, double* second_vector);                                   //
void* Routine_QR(void* rank);                                                                                            //
struct projects                                                                                                          //
{                                                                                                                        //
    double** projes;                                                                                                     //
    long j;                                                                                                              //
}Projs;                                                                                                                  //
                                                                                                                         //
void* Routine_proj(void* rank);                                                                                          //
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]) {
    /* Проверям, что параметры на месте.
    // Путь до 1 матрицы.
    // Путь до 2 матрицы.
    // Количество потоков.
    */
    if (argc < 3) {
        std::cout << "No requered params! Program will not start..." << std::endl;
        return -1;
    }

    // Получение матриц из файлов.
    long n_1, m_1, n_2, m_2;
    n_1 = m_1 = n_2 = m_2 = 0;
    long** matrix_1 = nullptr;
    long** matrix_2 = nullptr;
    try {
        matrix_1 = read_matrix(argv[1], n_1, m_1);
        matrix_2 = read_matrix(argv[2], n_2, m_2);
    } catch (const std::exception& e) {
            std::cout << "Ooops exception!";
            return -1;
    }


    // Считываем количество потоков для работы, если задано.
    if (argc >= 4)
        number_of_threads = strtol(argv[3], NULL, 10);
        
    // Выделяем память под потоки.
    pthread_t* thread_handles = (pthread_t*)malloc(number_of_threads * sizeof(pthread_t));

    // Инициализируем переменные для замера вермени.
    std::chrono::time_point<std::chrono::steady_clock> begin, end;

    // Проверка на то, что матрицы могут быть умножены. 
    if (m_1 != n_2) {
        std::cout << "Matrix sizes not valid! Work was stopped. Let's try QR decomposition..." << std::endl;
        clean_matrix(matrix_1, n_1);
        clean_matrix(matrix_2, n_2);
    } else {
        // Добавлям матрицы в структуру.
        matrix* matrix_struct_1 = (matrix*)malloc(sizeof(matrix));
        matrix* matrix_struct_2 = (matrix*)malloc(sizeof(matrix));

        matrix_struct_1->matrix = matrix_1;
        matrix_struct_1->n = n_1;
        matrix_struct_1->m = m_1;

        matrix_struct_2->matrix = matrix_2;
        matrix_struct_2->n = n_2;
        matrix_struct_2->m = m_2;

        // Создаём результирующую матрицу для построчного умножения.
        long** result_matrix = nullptr;
        result_matrix = (long**)calloc(n_1, sizeof(long*));
        for (long i = 0; i < n_1; ++i) {
            result_matrix[i] = (long*)calloc(m_2, sizeof(long));
        }

        // Создаём результирующую матрицу для блочного умножения.
        long** result_matrix_2 = nullptr;
        result_matrix_2 = (long**)calloc(n_1, sizeof(long*));
        for (long i = 0; i < n_1; ++i) {
            result_matrix_2[i] = (long*)calloc(m_2, sizeof(long));
        }

        // Создаём результрующую матрицу для блочного умножения.
        long** result_matrix_3 = nullptr;
        result_matrix_3 = (long**)calloc(n_1, sizeof(long*));
        for (long i = 0; i < n_1; ++i) {
            result_matrix_3[i] = (long*)calloc(m_2, sizeof(long));
        }

        // Добавляем аргументы для многопотчоного режима.
        Matrix_pair.matrix_1 = matrix_struct_1;
        Matrix_pair.matrix_2 = matrix_struct_2;
        Matrix_pair.result_matrix = result_matrix;


        // Измеряем время для умножения матриц по строкам. (Принято за идеал умножения по точности)
        begin = std::chrono::steady_clock::now();
        for(long thread = 0; thread < number_of_threads; ++thread)
            pthread_create(&thread_handles[thread], NULL, Routine_rows, (void*)thread);                          

        for(long thread = 0; thread < number_of_threads; ++thread)
        {
            pthread_join(thread_handles[thread], NULL);
        }
        end = std::chrono::steady_clock::now();
        std::cout << "The time rows: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms\n";

        // Меняем результирующую матрицу.
        Matrix_pair.result_matrix = result_matrix_2;

        // Создаём и инициализируем n_1 * m_2 мьютексов (с запасом) для борьбы с гонкой данных.
        mutexes = (pthread_mutex_t**)malloc(n_1 * sizeof(pthread_mutex_t*));
        for (int i = 0; i < n_1; ++i) {
            mutexes[i] = (pthread_mutex_t*)malloc(m_2 * sizeof(pthread_mutex_t));
            for (int j = 0; j < m_2; ++j) {
                pthread_mutex_init(&mutexes[i][j], NULL);
            }  
        }

        // Замеряем время для блочного уможния.
        begin = std::chrono::steady_clock::now();
        // Количество блоков == количество потоков + 1, но более чем число ячеек в матрице.
        // Умножение работает корректно только для квадратных матриц.
        // Доработка до полной произвольности может занять слишком много времени.
        if (n_1 == n_2 && n_1 == m_1 && m_1 == m_2) {
            long n_o_t = number_of_threads;
            if (number_of_threads > n_1 * m_1) n_o_t = n_1 * m_1;
            for(long thread = 0; thread < n_o_t; ++thread)
                pthread_create(&thread_handles[thread], NULL, Routine_blocks, (void*)thread);                          

            for(long thread = 0; thread < n_o_t; ++thread)
            {
                pthread_join(thread_handles[thread], NULL);
            }
        }
        else {
            std::cout << "Oooops problem whith matrix size for blocks methods!" << std::endl;
        }
        end = std::chrono::steady_clock::now();
        std::cout << "The time blocks: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms\n";

        // Меняем результрующую матрицу для умножения по столбцам.
        Matrix_pair.result_matrix = result_matrix_3;
        begin = std::chrono::steady_clock::now();
        for(long thread = 0; thread < number_of_threads; ++thread)
            pthread_create(&thread_handles[thread], NULL, Routine_columns, (void*)thread);                          

        for(long thread = 0; thread < number_of_threads; ++thread)
        {
            pthread_join(thread_handles[thread], NULL);
        }
        end = std::chrono::steady_clock::now();
        std::cout << "The time columns: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms\n";

        // Очищаем мьютексы.
        for (int i = 0; i < n_1; ++i) {
            for (int j = 0; j < m_2; ++j) {
                pthread_mutex_destroy(&mutexes[i][j]);
            }
            free(mutexes[i]);
        }
        free(mutexes);

        // Подсчёт количества ошибок между строчным методом и блочным.
        long error_count = 0;
        for (long i = 0; i < n_1; ++i) {
            for (long j = 0; j < m_2; ++j) {
                if (result_matrix[i][j] != result_matrix_2[i][j]) {
                    std::cout << result_matrix[i][j] << " vs " << result_matrix_2[i][j] << std::endl;
                    ++error_count;
                }
            }
        }
        std::cout << "Errors rows vs blocks: " << error_count << ", From: " << n_1*m_2 << std::endl;

        // Подсчёт количества ошибок между строчным методом и столбчатным.
        error_count = 0;
        for (long i = 0; i < n_1; ++i) {
            for (long j = 0; j < m_2; ++j) {
                if (result_matrix[i][j] != result_matrix_3[i][j]) {
                    ++error_count;
                }
            }
        }
        std::cout << "Errors rows vs columns: " << error_count << ", From: " << n_1*m_2 << std::endl;

        //print_matrix(result_matrix_2, n_1, m_2);

        // Очищаем матрицы.
        clean_matrix(result_matrix, n_1);
        clean_matrix(result_matrix_2, n_1);
        clean_matrix(result_matrix_3, n_1);

        //free(matrix_struct_1);
        //free(matrix_struct_2);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // QR разложение. //
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    // Определяем точность вывода.
    std::cout << std::fixed << std::setprecision(2);

    // Переводим long матрицу в double.
    double** double_matrix;
    double_matrix = (double**)matrix_1;
    for (long i = 0; i < n_1; ++i) {
        double_matrix[i] = (double*)matrix_1[i];
        for (long j = 0; j < m_1; ++j) {
            double_matrix[i][j] = (double)matrix_1[i][j];
        }
    }

    // Создадим и инициализируем структуру для многопоточности.
    matrix_double* vectors = (matrix_double*)malloc(sizeof(double_matrix));
    vectors->matrix = double_matrix;
    vectors->n = n_1;
    vectors->m = m_1;

    // Инициализируем симафоры.
    sem_init(&semaphore, 0, 0);
    sem_init(&semaphore_2, 0, 0);

    // Запускаем функцию для QR разложения с одной критичной зоной.
    begin = std::chrono::steady_clock::now();
    QR(vectors, thread_handles);
    end = std::chrono::steady_clock::now();

    std::cout << "The time QR: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms\n";

    // Создание матрицы, которая должна быть идентична входящей для проверки разложения A=QR.
    double** test_matrix = (double**)calloc(vectors->n, sizeof(double*));
    for (long i = 0; i < vectors->n; ++i) {
        test_matrix[i] = (double*)calloc(vectors->m, sizeof(double));
    }

    // QR умножение построчно.
    for(long row_1 = 0; row_1 < vectors->m; ++row_1) 
        for(long column_2 = 0; column_2 < vectors->m; ++column_2) 
            for(long column_1 = 0; column_1 < vectors->n; ++column_1) {
                test_matrix[row_1][column_2] += QR_matrix.Q[row_1][column_1] * QR_matrix.R[column_1][column_2];
            }

    std::cout << "TEST matrix" << std::endl;
    // Вывод полученной матрицы.
    //print_matrix(test_matrix, vectors->m, vectors->m);

    // Вывед QR матриц.
    //print_matrix(QR_matrix.Q, n_1, m_1);
    //print_matrix(QR_matrix.R, n_1, m_1);

    // Очиаем семафоры.
    sem_destroy(&semaphore);
    sem_destroy(&semaphore_2);
    
    // Очищаем матрицы.
    clean_matrix(matrix_1, n_1);
    clean_matrix(matrix_2, n_2);

    //clean_matrix(QR_matrix.Q, n_1);
    //clean_matrix(QR_matrix.R, n_1);

    clean_matrix(test_matrix, n_1);

    //clean_matrix(double_matrix, n_1);
    free(thread_handles);
    free(vectors);
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

void clean_matrix(double** matrix, const long n) {
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

const void print_matrix(double** matrix, const long n, const long m) {
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

// Обычное умножение матриц по строкам, с указанием конкретных строк и солбцов участвующиз в умножении.
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

// Умножение по столбцам, также с указанием конкртеных границ. Используется синхронизация с помощью мьютексов.
// Происходит блокировка всей строки, так как мы взаимодействуем именно с ней.
void sum_matrix_columns(long** matrix_1, long** matrix_2,
                     long column_1_start, long row_1_start,
                     long column_2_start, long column_1_end,
                     long row_1_end, long column_2_end,
                     long** result_matrix) {
    
    for(long column_1 = column_1_start; column_1 < column_1_end && column_1 < Matrix_pair.matrix_1->m; ++column_1) {
        for(long row_1 = row_1_start; row_1 < row_1_end && row_1 < Matrix_pair.matrix_1->n; ++row_1) {
             //pthread_mutex_lock(&mutexes[row_1][0]);
             for(long column_2 = column_2_start; column_2 < column_2_end && column_2 < Matrix_pair.matrix_2->m; ++column_2) {
                result_matrix[row_1][column_2] += matrix_1[row_1][column_1] * matrix_2[column_1][column_2];
             }
             //pthread_mutex_unlock(&mutexes[row_1][0]);
        }
    }
}

// Блочное умножение матриц, после дополнительных тестов, мьютексы оказались не нужны.
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

// Распределение строк между потоками.
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

// Распределение столбцов между потоками
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

void* Routine_blocks(void* rank) {
    long current_rank = (long)rank;

    // Определеяем средний размер блока.

    long block_size = long(sqrt(double(number_of_threads)));
    long block_size_n = Matrix_pair.matrix_1->n / block_size;
    long block_size_m = Matrix_pair.matrix_1->m / block_size;

    if (!block_size_n) block_size_n = 1;
    if (!block_size_m) block_size_m = 1;


    // Определяем стартовую позицию для конкретного блока.
    long block_row_start = 0;
    long block_column_start = 0;
    long block_common_index = 0;

    position* current_position = get_position(current_rank, block_size_m, block_size_n);
    block_row_start = current_position->x;
    block_column_start = current_position->y;
    free(current_position);

    // Выполняем расчёты для блоков.
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


    // Определяем позицю последнего блока.
    current_position = get_position(number_of_threads-1, block_size_m, block_size_n);
    block_row_start = current_position->x;
    block_column_start = current_position->y;

    free(current_position);

    // Расчитываем количество строк для досчёта n % block_size_n, определяем их стратовую позицию.
    long rows = Matrix_pair.matrix_1->n - (block_row_start + block_size_n + 1);
    int number_of_rows = rows / number_of_threads;
    // Распределяем строки  между потоками.
    long first_row = (block_row_start + block_size_n) + current_rank * number_of_rows;
    long last_row = (block_row_start + block_size_n) + (current_rank + 1) * number_of_rows;

    if (current_rank == (number_of_threads-1) && last_row < Matrix_pair.matrix_1->n)
        last_row = Matrix_pair.matrix_1->n;

    // Досчитываем оставшиеся пропуски, ЕСЛИ ЭТО НЕОБХОДИМО, иначе не пропустит проверка границ.
    sum_matrix_rows(Matrix_pair.matrix_1->matrix, Matrix_pair.matrix_2->matrix, first_row, 0, 0, last_row,
                    Matrix_pair.matrix_1->m, Matrix_pair.matrix_2->m, Matrix_pair.result_matrix);


    // Определяем столбцы m % block_size_m и распределяем между потоками.
    long column_index = Matrix_pair.matrix_1->m -(Matrix_pair.matrix_1->m % block_size_m);
    long columns = Matrix_pair.matrix_1->m -(column_index + 1);

    int number_of_columns = columns / number_of_threads;
    if (number_of_columns == 0) number_of_columns = 1;
    long first_column = column_index + current_rank * number_of_columns;
    long last_column = column_index + (current_rank + 1) * number_of_columns;

    if (current_rank == (number_of_threads-1)) //&& last_column < Matrix_pair.matrix_1->m)
        last_column = Matrix_pair.matrix_1->m;

    sum_matrix_columns(Matrix_pair.matrix_1->matrix, Matrix_pair.matrix_2->matrix, 0, 0, first_column, Matrix_pair.matrix_1->m, first_row, last_column, Matrix_pair.result_matrix);

    rows = block_size_n; 
    number_of_rows = rows / number_of_threads;
    if (number_of_rows == 0)
        number_of_rows = 1;

    long save_first_row = first_row;
    first_row = block_row_start + current_rank * number_of_rows;
    last_row = block_row_start + (current_rank + 1) * number_of_rows;

    if (current_rank == (number_of_threads - 1) && last_row < save_first_row)
        last_row = save_first_row;

    // Перемножаем построчно элементы оставшиеся на месте пустых блоков.
    if (first_row < save_first_row)
        sum_matrix_rows(Matrix_pair.matrix_1->matrix, Matrix_pair.matrix_2->matrix, first_row, 0, block_column_start + block_size_m, last_row, Matrix_pair.matrix_1->m, column_index, Matrix_pair.result_matrix);

    // Умножаем постолбцам элементы оставшиеся между блокам.
    // number_of_columns = Matrix_pair.matrix_1->m / number_of_threads;
    // first_column = current_rank * number_of_columns;
    // last_column = (current_rank + 1) * number_of_columns;

    // if (first_row < save_first_row)
    //     sum_matrix_columns(Matrix_pair.matrix_1->matrix, Matrix_pair.matrix_2->matrix, first_column, block_row_start, block_column_start + block_size_m, last_column, save_first_row, column_index, Matrix_pair.result_matrix);


    return NULL;
}


// Получение позиции для блока.
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


////////
// QR //
////////

// Скалярное умножение векторов.
double column_scalar_multiplication(matrix_double* vectors, long column_i, matrix_double* vectors_second, long column_j) {
    double answer = 0;
    for (long i = 0; i < vectors->n; i++) {
        answer += vectors->matrix[i][column_i] * vectors_second->matrix[i][column_j];
    }
    return answer;
}

// Вычисление проекции вектора.
double* column_proj(matrix_double* b, matrix_double* a, long column_i, long column_j) {
    double cf = column_scalar_multiplication(a, column_j, b, column_i) / column_scalar_multiplication(b, column_i, b, column_i);
    double* proj = (double*)calloc(b->n, sizeof(double));
    for (long i = 0; i < b->n; ++i) {
        proj[i] = cf * b->matrix[i][column_i];
    }
    return proj;
}

// QR разложение
void QR(matrix_double* vectors, pthread_t* thread_handles) {

    // Создаём b матрицу, как копию a
    matrix_double* updated_matrix = (matrix_double*)malloc(sizeof(matrix_double));
    updated_matrix->n = vectors->n;
    updated_matrix->m = vectors->m;
    updated_matrix->matrix = (double**)malloc(vectors->n * sizeof(double*));
    for (long i = 0; i < updated_matrix->n; ++i) {
        updated_matrix->matrix[i] = (double*)malloc(vectors->m * sizeof(double));
    }

    Matrix_for_QR.matrix_1 = vectors;
    Matrix_for_QR.matrix_2 = updated_matrix;

    // Задём указатель на b матрицу, как q матрицу. 
    QR_matrix.Q = updated_matrix->matrix;

    // Выделяем память под T матрицу.
    matrix_double* Q_t = (matrix_double*)malloc(sizeof(matrix_double));
    Q_t->matrix = (double**)malloc(updated_matrix->m*sizeof(double*));
    for (long i = 0; i < updated_matrix->m; ++i) {
        Q_t->matrix[i] = (double*)malloc(updated_matrix->n*sizeof(double));
    }

    // Заполняем структуры с матрциами.
    Matrixes_Q.matrix_1 = updated_matrix;
    Matrixes_Q.matrix_1->n = updated_matrix->n;
    Matrixes_Q.matrix_1->m = updated_matrix->m;
    Matrixes_Q.matrix_2 = Q_t;
    Matrixes_Q.matrix_2->n = updated_matrix->m;
    Matrixes_Q.matrix_2->m = updated_matrix->n;

    // Выделяем память по R матрицу.
    QR_matrix.R = (double**)calloc(updated_matrix->n, sizeof(double*));
    for (long i = 0; i < updated_matrix->n; ++i) {
        QR_matrix.R[i] = (double*)calloc(updated_matrix->m, sizeof(double));
    }

    // Запускаем потоки для части вычислений.
    for(long thread = 0; thread < number_of_threads; ++thread)
        pthread_create(&thread_handles[thread], NULL, Routine_QR, (void*)thread);                          

    // Ожидаем заверешения копирования матрицы.
    int counter = 0;
    for (int i = 0; i < number_of_threads; ++i) {
        sem_wait(&semaphore);
    }

    // Высчитываем проекции паралельно (пока работает не стабильно) и выполняем последовательные вычитания.
    //pthread_t* thread_handlers = (pthread_t*)malloc(number_of_threads * sizeof(pthread_t));
    //Projs.projes = (double**)calloc((vectors->m - 1), sizeof(double*));
    for (long j = 0; j < vectors->m; ++j) {
        //Projs.j = j;
        //for(long thread = number_of_threads; thread < number_of_threads + number_of_threads; ++thread)
            //pthread_create(&thread_handlers[thread], NULL, Routine_proj, (void*)thread);                          

        //for(long thread = number_of_threads; thread < number_of_threads + number_of_threads; ++thread)
        //{
        //    pthread_join(thread_handlers[thread], NULL);
        //}
        for (long i = 0; i < j; ++i) {
            //double* proj = Projs.projes[i];
            double* proj = column_proj(Matrix_for_QR.matrix_2, Matrix_for_QR.matrix_1, i, j);
            vector_subtraction(updated_matrix, j, proj);
            free(proj);
        }
    }
    //free(thread_handlers);
    //free(Projs.projes);
    //print_matrix(Matrix_for_QR.matrix_2->matrix, Matrix_for_QR.matrix_2->n, Matrix_for_QR.matrix_2->m);

    // Возобновляем работу потоков.
    for (int i = 0; i < number_of_threads; ++i) {
        sem_post(&semaphore_2);
    } 
    for(long thread = 0; thread < number_of_threads; ++thread)
    {
        pthread_join(thread_handles[thread], NULL);
    }
}

// Вычитания векторов.
void vector_subtraction(matrix_double* vectors, long column_i, double* second_vector) {
    for (long i = 0; i < vectors->n; ++i) {
        vectors->matrix[i][column_i] -= second_vector[i];
        //std::cout << vectors->matrix[i][column_i] << std::endl;
    }
}


// Паралельный алгоритм для QR матрицы.
void* Routine_QR(void* rank) {

    long current_rank = (long)rank;

    // Определяем количество строчек для каждоого потока.
    long line_size = Matrix_for_QR.matrix_2->n / number_of_threads;
    if (line_size == 0) line_size = 1;
    long start_line = current_rank * line_size;
    long end_line = (current_rank + 1) * line_size;

    if (current_rank == number_of_threads - 1)
        end_line = Matrix_for_QR.matrix_2->n;

    // Копируем матрицу a в матрицу b.
    for (long i = start_line; i < end_line && i < Matrix_for_QR.matrix_2->n; ++i)
        for (long j = 0; j < Matrix_for_QR.matrix_2->m && j < Matrix_for_QR.matrix_2->m; ++j)  {
            Matrix_for_QR.matrix_2->matrix[i][j] = Matrix_for_QR.matrix_1->matrix[i][j];
        }

    // Ожидаем заверешения расчётов проекций.
    sem_post(&semaphore);
    sem_wait(&semaphore_2);

    // Определяем размер столбца для каждого потока.
    long column_size = Matrix_for_QR.matrix_2->m / number_of_threads;
    if (column_size == 0) column_size = 1;
    long start_column = current_rank * column_size;
    long end_column = (current_rank + 1) * column_size;

    if (current_rank == number_of_threads - 1)
        end_column = Matrix_for_QR.matrix_2->m;

    // Нормируем матрицу.
    for (long j = start_column; j < end_column && j < Matrix_for_QR.matrix_2->m; ++j) {
        double cf = sqrt(column_scalar_multiplication(Matrix_for_QR.matrix_2, j, Matrix_for_QR.matrix_2, j));
        for (long i = 0; i < Matrix_for_QR.matrix_2->n; ++i) {
            Matrix_for_QR.matrix_2->matrix[i][j] /= cf;
        }
    }

    // Расчитываем столбцы.
    column_size = Matrixes_Q.matrix_2->m / number_of_threads;
    start_column = current_rank * column_size;
    end_column = (current_rank + 1) * column_size;

    if (current_rank == number_of_threads - 1)
        end_column = Matrixes_Q.matrix_2->m;

    // Транспонируем матрицу.
    for (long j = start_column; j < end_column && j < Matrix_for_QR.matrix_2->m; ++j) {
        for (long i = 0; i < Matrixes_Q.matrix_2->n; ++i) {
            Matrixes_Q.matrix_2->matrix[j][i] = Matrixes_Q.matrix_1->matrix[i][j];
        }
    }

    // Построчно паралельно умножаем матрицу QT * A, чтобы получить R матрицу.
    for(long row_1 = start_column; row_1 < end_column && row_1 < Matrixes_Q.matrix_2->m; ++row_1) 
        for(long column_2 = 0; column_2 < Matrix_for_QR.matrix_1->m; ++column_2) 
            for(long column_1 = 0; column_1 < Matrixes_Q.matrix_2->m; ++column_1) {
                QR_matrix.R[row_1][column_2] += Matrixes_Q.matrix_2->matrix[row_1][column_1] * Matrix_for_QR.matrix_1->matrix[column_1][column_2];
            }

    return NULL;
}

// Расчёт проекций и запись их в вектор, вектров проекций.
void* Routine_proj(void* rank) {
    long current_rank = (long)rank - number_of_threads;
    long i_blocks = Projs.j / number_of_threads;
    if (i_blocks == 0) i_blocks = 1;
    long start_i = current_rank * i_blocks;
    long end_i = (current_rank + 1) * (i_blocks);
    if (current_rank == number_of_threads-1) {
        end_i = Projs.j;
    }
    for (long i = start_i; i < end_i; ++i) {
        Projs.projes[i] = column_proj(Matrix_for_QR.matrix_2, Matrix_for_QR.matrix_1, i, Projs.j);
    }
    return NULL;
}