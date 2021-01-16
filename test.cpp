#include <iostream>
#include <random>
#include <math.h>

typedef struct
{
    int x;
    int y;
    int z;

} dim3;

void print_matrix(unsigned int *mat, int n_rows, int n_cols)
{
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            std::cout << mat[i * n_cols + j] << ", ";
        }
        std::cout << std::endl;
    }
}

void cross(

    dim3 blockIdx,
    dim3 blockDim,
    dim3 threadIdx,

    unsigned int *population,
    unsigned int *match,
    unsigned int *born,

    int n_rows,
    int n_cols
)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n_rows || n_cols < 2)
    {
        return;
    }

    int mid_point = 0;
    if (n_cols % 2 == 0)
    {
        mid_point = (n_cols / 2) - 1;
    }
    else
    {
        mid_point = (n_cols-1) / 2;
    }

    if (threadIdx.x == 0)
    {
        int i = 0;
        while (i <= mid_point)
        {
            born[2 * row * n_cols + i] = population[row * n_cols + i];
            i = i + 1;
        }

        while (i <= (n_cols-1))
        {
            born[2 * row * n_cols + i] = population[match[row] * n_cols + i];
            i = i + 1;
        }
    }
    else if (threadIdx.x == 1)
    {
        int i = 0;
        while (i <= mid_point)
        {
            born[(2*row + 1) * n_cols + i] = population[match[row] * n_cols + i];
            i = i + 1;
        }

        while (i <= (n_cols-1))
        {
            born[(2*row + 1) * n_cols + i] = population[row * n_cols + i];
            i = i + 1;
        }
    }
    else
    {
        return;
    }
}

int main()
{
    int n_rows = 6;
    int n_cols = 8;
    unsigned int *data = (unsigned int *) malloc(n_rows * n_cols * sizeof(unsigned int));
    unsigned int *match = (unsigned int *) malloc(n_rows * sizeof(unsigned int));
    unsigned int *born = (unsigned int *) malloc(2 * n_rows * n_cols * sizeof(unsigned int));

    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned int> distribution(0, n_rows-1);

    for (int i = 0; i < n_rows; i++)
    {
        match[i] = distribution(generator);

        for (int j = 0; j < n_cols; j++)
        {
            data[i * n_cols + j] = distribution(generator);
        }
    }
    
    for (int i = 0; i < n_rows; i++)
    {
        for (int j = 0; j < n_cols; j++)
        {
            born[2 * i * n_cols + j] = 0;
            born[(2 * i + 1) * n_cols + j] = 0;
        }
    }

    // print_matrix(data, n_rows, n_cols);
    // std::cout << std::endl;
    // print_matrix(match, n_rows, 1);
    // std::cout << std::endl;
    // print_matrix(born, 2*n_rows, n_cols);
    // std::cout << std::endl;

    // std::cout << "开始处理！" << std::endl << std::endl;

    const int BLOCK_SIZE = 16;

    dim3 blockDim;
    blockDim.x = 2;
    blockDim.y = BLOCK_SIZE;
    blockDim.z = 1;

    dim3 gridDim;
    gridDim.x = 1;
    gridDim.y = (unsigned int) ceil((((double) n_rows)+0.0000001)/blockDim.y);
    gridDim.z = 1;

    dim3 threadIdx;
    dim3 blockIdx;

    for (blockIdx.x = 0; blockIdx.x < gridDim.x; blockIdx.x = blockIdx.x + 1)
    {
        for (blockIdx.y = 0; blockIdx.y < gridDim.y; blockIdx.y = blockIdx.y + 1)
        {
            for (threadIdx.x = 0; threadIdx.x < blockDim.x; threadIdx.x = threadIdx.x + 1)
            {
                for (threadIdx.y = 0; threadIdx.y < blockDim.y; threadIdx.y = threadIdx.y + 1)
                {

                    // std::cout << "t: " << "( " << threadIdx.x << ", " << threadIdx.y << " )" << std::endl;
                    // std::cout << "b: " << "( " << blockIdx.x << ", " << blockIdx.y << " )" << std::endl;

                    cross(
                        blockIdx, 
                        blockDim, 
                        threadIdx,
                        data,
                        match,
                        born,
                        n_rows,
                        n_cols
                    );
                }
            }
        }
    }

    print_matrix(data, n_rows, n_cols);
    std::cout << std::endl;
    print_matrix(match, n_rows, 1);
    std::cout << std::endl;
    print_matrix(born, 2*n_rows, n_cols);
    std::cout << std::endl;

    free(data);
    free(match);
    free(born);
}