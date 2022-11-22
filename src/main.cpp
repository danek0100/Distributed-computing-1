#include <iostream>
#include <fstream>

int32_t** read_matrix(char* path, int32_t& n, int32_t& m);
void clean_matrix(int32_t** matrix, const int32_t n);
const void print_matrix(int32_t** matrix, const int32_t n, const int32_t m);
int32_t** sum_matrix_rows(int32_t** matrix_1, const int32_t n_1, const int32_t m_1,
                          int32_t** matrix_2, const int32_t n_2, const int32_t m_2);

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
    print_matrix(matrix_2, n_1, m_2);

    int32_t** result_matrix = nullptr;
    try
    {
       result_matrix = sum_matrix_rows(matrix_1, n_1, m_1, matrix_2, n_2, m_2);
    } catch(const int e) {
        std::cout << "Matrix sizes not valid! Work was stopped." << std::endl;
        return -1;
    }

    print_matrix(result_matrix, m_1, n_2);
    
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
                          int32_t** matrix_2, const int32_t n_2, const int32_t m_2) {
    
    if (m_1 != n_2) {
        throw 1;
    }

    int32_t** result_matrix = nullptr;    
    result_matrix = (int32_t**)malloc(sizeof(int32_t*) * m_1);
    for (int32_t i = 0; i < m_1; ++i) {
        result_matrix[i] = (int32_t*)calloc(n_2, sizeof(int32_t));
    }

    for(int32_t row_1 = 0; row_1 < n_1; ++row_1) 
        for(int32_t column_2 = 0; column_2 < m_2; ++column_2) 
            for(int32_t column_1 = 0; column_1 < m_1; ++column_1)
                result_matrix[row_1][column_2] += matrix_1[row_1][column_1] * matrix_2[column_1][column_2];

    return result_matrix;
}