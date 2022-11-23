#include <iostream>
#include <fstream>

int32_t** read_matrix(char* path, int32_t& n, int32_t& m);
void clean_matrix(int32_t** matrix, const int32_t n);
const void print_matrix(int32_t** matrix, const int32_t n, const int32_t m);
int32_t** sum_matrix_rows(int32_t** matrix_1, const int32_t n_1, const int32_t m_1,
                          int32_t** matrix_2, const int32_t n_2, const int32_t m_2,
                          int32_t row_1_start=0, int32_t column_1_start=0,
                          int32_t column_2_start=0, int32_t row_1_end=0,
                          int32_t column_1_end=0, int32_t column_2_end=0,
                          int new_matrix=1, int32_t** result_matrix = nullptr);
                          
int32_t** sum_matrix_columns(int32_t** matrix_1, const int32_t n_1, const int32_t m_1,
                             int32_t** matrix_2, const int32_t n_2, const int32_t m_2);

int32_t** sum_matrix_blocks(int32_t** matrix_1, const int32_t n_1, const int32_t m_1,
                            int32_t** matrix_2, const int32_t n_2, const int32_t m_2, int blocks_num=1);

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "No requered params! Program go to rest." << std::endl;
        return -1;
    }

    int32_t n_1, m_1, n_2, m_2;
    n_1 = m_1 = n_2 = m_2 = 0;
    int32_t** matrix_1 = nullptr;
    int32_t** matrix_2 = nullptr;
    try {
        matrix_1 = read_matrix(argv[1], n_1, m_1);
        matrix_2 = read_matrix(argv[2], n_2, m_2);
    } catch(const std::exception& e) {
            std::cout << "Ooops exception!";
            return -1;
    }

    long number_of_threads = 1;
    if (argc >= 4)
        number_of_threads = strtol(argv[3], NULL, 10);

    print_matrix(matrix_1, n_1, m_1);
    print_matrix(matrix_2, n_2, m_2);

    int32_t** result_matrix = nullptr;
    try
    {
        result_matrix = sum_matrix_rows(matrix_1, n_1, m_1, matrix_2, n_2, m_2);
    } catch(const int e) {
        std::cout << "Matrix sizes not valid! Work was stopped." << std::endl;
        return -1;
    }

    print_matrix(result_matrix, n_1, m_2);
    clean_matrix(result_matrix, m_1);

    try
    {
        result_matrix = sum_matrix_blocks(matrix_1, n_1, m_1, matrix_2, n_2, m_2, 2);
    } catch(const int e) {
        std::cout << "Matrix sizes not valid! Work was stopped." << std::endl;
        return -1;
    }

    print_matrix(result_matrix, n_1, m_2);
    clean_matrix(result_matrix, m_1);


    try
    {
       result_matrix = sum_matrix_columns(matrix_1, n_1, m_1, matrix_2, n_2, m_2);
    } catch(const int e) {
        std::cout << "Matrix sizes not valid! Work was stopped." << std::endl;
        return -1;
    }


    

    print_matrix(result_matrix, n_1, m_2);
    
    clean_matrix(matrix_1, n_1);
    clean_matrix(matrix_2, n_2);
    clean_matrix(result_matrix, m_1);

    std::cout << "Hi, gays! I want to get " << number_of_threads << "$!" << std::endl;
    return 0;
}

int32_t** read_matrix(char* path, int32_t& n, int32_t& m) {
    int32_t** matrix = nullptr;
    std::ifstream in(path);
    if (in.is_open())
    {
        try {
            in >> n >> m;
        } catch (const std::exception& e) {
            std::cerr << "N and M parsing error: " << e.what() << std::endl;
            throw e;
        }
        matrix = (int32_t**)malloc(sizeof(int32_t*) * n);
        for (int32_t i = 0; i < n; ++i) {
            matrix[i] = (int32_t*)calloc(m, sizeof(int32_t));
        }
        try
        {
            for (int32_t i = 0; i < n; i++)
                for (int32_t j = 0; j < m; ++j)
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

void clean_matrix(int32_t** matrix, const int32_t n) {
    for (int32_t i = 0; i < n; ++i)
        free(matrix[i]);
    free(matrix);   
}

const void print_matrix(int32_t** matrix, const int32_t n, const int32_t m) {
    std::cout << "==================================================" << std::endl;
    for (int32_t i = 0; i < n; ++i) {
        for (int32_t j = 0; j < m; ++j)
            std::cout << matrix[i][j] << '\t';
        std::cout << std::endl;
    }
    std::cout << "==================================================" << std::endl;
        
}

int32_t** sum_matrix_rows(int32_t** matrix_1, const int32_t n_1, const int32_t m_1,
                          int32_t** matrix_2, const int32_t n_2, const int32_t m_2,
                          int32_t row_1_start, int32_t column_1_start,
                          int32_t column_2_start, int32_t row_1_end,
                          int32_t column_1_end, int32_t column_2_end,
                          int new_matrix, int32_t** result_matrix) {
    
    if (m_1 != n_2) {
        throw 1;
    }

    if (result_matrix == nullptr) {
        row_1_end = n_1;
        column_1_end = m_1;
        column_2_end = m_2;
    }

    if (new_matrix) {
        result_matrix = (int32_t**)malloc(sizeof(int32_t*) * m_1);
        for (int32_t i = 0; i < m_1; ++i) {
            result_matrix[i] = (int32_t*)calloc(n_2, sizeof(int32_t));
        }
    }


    for(int32_t row_1 = row_1_start; row_1 < row_1_end; ++row_1) 
        for(int32_t column_2 = column_2_start; column_2 < column_2_end; ++column_2) 
            for(int32_t column_1 = column_1_start; column_1 < column_1_end; ++column_1) {
                result_matrix[row_1][column_2] += matrix_1[row_1][column_1] * matrix_2[column_1][column_2];
            }

    return result_matrix;
}

int32_t** sum_matrix_columns(int32_t** matrix_1, const int32_t n_1, const int32_t m_1,
                             int32_t** matrix_2, const int32_t n_2, const int32_t m_2) {
    
    if (m_1 != n_2) {
        throw 1;
    }

    int32_t** result_matrix = nullptr;
    result_matrix = (int32_t**)malloc(sizeof(int32_t*) * m_1);
    for (int32_t i = 0; i < m_1; ++i) {
        result_matrix[i] = (int32_t*)calloc(n_2, sizeof(int32_t));
    }

    for(int32_t column_1 = 0; column_1 < m_1; ++column_1)
        for(int32_t row_1 = 0; row_1 < n_1; ++row_1)
            for(int32_t column_2 = 0; column_2 < m_2; ++column_2) 
                result_matrix[row_1][column_2] += matrix_1[row_1][column_1] * matrix_2[row_1][column_2];

    return result_matrix;
}

int32_t** sum_matrix_blocks(int32_t** matrix_1, const int32_t n_1, const int32_t m_1,
                            int32_t** matrix_2, const int32_t n_2, const int32_t m_2, int blocks_num) {
    
    if (m_1 != n_2) {
        throw 1;
    }

    int32_t** result_matrix = nullptr;
    result_matrix = (int32_t**)malloc(sizeof(int32_t*) * m_1);
    for (int32_t i = 0; i < m_1; ++i) {
        result_matrix[i] = (int32_t*)calloc(n_2, sizeof(int32_t));
    }

    int32_t* boarders_row_1 = (int32_t*)calloc(blocks_num+1, sizeof(int32_t));
    int32_t* boarders_column_1 = (int32_t*)calloc(blocks_num+1, sizeof(int32_t));
    int32_t* boarders_column_2 = (int32_t*)calloc(blocks_num+1, sizeof(int32_t));
    int i = 0;
    int32_t board = 0;
    int32_t block_row_1_size = n_1 / blocks_num;
    int32_t block_column_1_size = m_1 / blocks_num;
    int32_t block_column_2_size = m_2 / blocks_num;

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



    for (int i = 0; i <= blocks_num; ++i) {
        for (int j = 0; j <= blocks_num; ++j) {
            for (int c = 0; c <= blocks_num; ++c) {
                result_matrix = sum_matrix_rows(matrix_1, n_1, m_1, matrix_2, n_2, m_2, boarders_row_1[i], boarders_column_1[j], boarders_column_2[c],
                                                boarders_row_1[i+1], boarders_column_1[j+1], boarders_column_2[c+1], 0, result_matrix); 
            }
        }
    }

    return result_matrix;
}