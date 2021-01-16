import cupy as cp

# 输入：
# in_routes: 每一行对应一个 route ，每一个 route 是 vertex 的列表
# in_distance_mat: 它的 [i, j] 元素 是 vertex i 到 vertex j 的距离
# problem_size: in_routes 的列维数
# pop_size: in_routes 的行维数
# 
# 输出：
# out_distance: 是输出值，对应 in_routes 每一行所对应的 route 的距离
distance_kernel = cp.RawKernel(
    r'''
    extern "C" __global__
    void distance(
        unsigned int *in_routes, 
        
        float *in_distance_mat, 
        float *out_distances,
        
        int problem_size,
        int pop_size
    )
    {   
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (row >= pop_size || col >= problem_size)
        {
            return;
        }
        
        int from = in_routes[row * problem_size + col];
        int to = in_routes[row * problem_size + 0];
        
        if (col < problem_size - 1)
        {
            to = in_routes[row * problem_size + col + 1];
        }
        
        out_distances[row * problem_size + col] = in_distance_mat[from * problem_size + to];
    }
    ''',
    'distance'
)

# 对于 i = 1, 2, ..., n_rows, 
# 将 data[i, col_ind[i]] 的 bit_ind[i] 位翻转
# n_rows 是 data 的行维数，n_cols 是 data 的列维数
# mutate[i] 取 0 表示 data[i] 不参与位翻转，取 1 表示参与
bit_flop_kernel = cp.RawKernel(
    r'''
    extern "C" __global__
    void bit_flop(
        unsigned int *data, 
        int n_rows,
        int n_cols,
        
        unsigned int *col_ind, 
        unsigned int *bit_ind,

        unsigned short *mutate
    )
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (row >= n_rows || col >= 1)
        {
            return;
        }

        if (mutate[row] == 0)
        {
            return;
        }
        
        unsigned int mask = 1;
        mask = mask << bit_ind[row];
        data[row * n_cols + col_ind[row]] ^= mask;
        
    }
    ''',
    'bit_flop'
)

# 同时让每一个 population 开始 cross
# population 是要参与 cross 的，每一行对应一个个体
# match 是一个 pop_size 行的向量，
# 并且 population[i] 将与 population[match[i]] 进行 cross
# 让 population[i] 的左半部分与 population[match[i]] 的右半部分拼接，得一个后代
# 再让 population[i] 的右半部分与 population[match[i]] 的左半部分拼接，得一个后代
# born 是 2 * pop_size 行的，列维数与 population 的相同也是 n_cols
cross_kernel = cp.RawKernel(
    r'''
    extern "C" __global__
    void cross(
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
    ''',
    'cross'
)