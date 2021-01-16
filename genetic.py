import cupy as cp

from math import ceil

from cuda_kernels import distance_kernel
from cuda_kernels import bit_flop_kernel
from cuda_kernels import cross_kernel

class Population:

    # 每个个体的基因型
    genes: cp.ndarray

    # 每个个体的表现型
    phenotype: cp.ndarray

    # 每个个体的适应度得分（对环境适应越好，分越高）
    scores: cp.ndarray

    # 每个个体有多少个基因位点
    n_genes: int

    # 种群的规模（有多少个个体）
    n_size: int

    def __init__(self, genes: cp.ndarray) -> None:
        self.genes = genes
        self.n_genes = genes.shape[1]
        self.n_size = genes.shape[0]

    # 生成初始种群
    @classmethod
    def generate_initial_population(cls, n_size: int, n_genes: int):

        genes = cp.random.randint(
            low = 0, 
            high = n_genes,
            size = (n_size, n_genes, ), 
            dtype = cp.uint32
        )

        return Population(genes)
    
    # 从基因型计算出表现型
    def calculate_phenotype(self) -> None:
        self.phenotype = cp.argsort(self.genes, axis = 1).astype(cp.uint32)

    # 计算得分
    def calculate_scores(self) -> None:

        # 将 genes (基因型) 转化为 phenotype (表现型)
        # routes 的每行是一个 route，一个 route 是一个关于 vertex 的列表
        routes = self.phenotype

        # distance_mat[i, j] 决定了 vertex i -> vertex j 的距离
        # 这个届时换成特定的距离矩阵
        distance_mat = cp.random.rand(self.n_genes, self.n_genes, dtype = cp.float32)

        # 确定要调用的 GPU 资源的规模，参看 CUDA Programming Guide
        # out_distances[i, j] 代表 routes[i] 的从 routes[i, j] 走到 routes[i, j+1] 的距离
        # 如果 j+1 == routes.shape[1]，则 out_distances[i, j] = 距离[routes[i, j], routes[i, 0]]
        BLOCK_SIZE = 16
        dim_block = (BLOCK_SIZE, BLOCK_SIZE, )
        dim_grid = (ceil(self.n_genes/dim_block[1]), ceil(self.n_size/dim_block[0]), )
        out_distances = cp.zeros(shape = self.genes.shape, dtype = cp.float32)
        distance_kernel(
            dim_grid, 
            dim_block, 
            (routes, distance_mat, out_distances, self.n_genes, self.n_size,)
        )

        # distances_per_sample[i] 是 routes[i] 走过的总距离，
        # 也就是，从 routes[i, 0] 出发，再回到 routes[i, 0] 走过的总距离
        distances_per_sample = cp.sum(out_distances, axis=1)

        # 距离越长，得分越低
        self.scores = cp.subtract(0, distances_per_sample)
    
    # 计算每个个体的生存几率
    def calculate_survival_chance(self):

        # 首先获取适应度，值越高，适应得越好
        scores = self.scores

        # 升序排列，数字越高，对应的适应度越高
        order = cp.argsort(cp.argsort(scores))

        # 把排位改变为降序排列
        order = cp.subtract(self.n_size - 1, order)

        # probs[i] 是 population[i] 被选入下一代的概率（没被选入的则被淘汰）
        probs = cp.divide(order, cp.sum(order))

        self.probs = probs

    # 通过基因变异引入新性状
    def populations_mutate(self) -> None:

        # 随机确定选哪些个 population 进行 mutate
        # 如果 mutate[i] == 1，则表示要对 self.genes[i] 进行 mutate 操作
        mutate = cp.random.randint(
            low = 0,
            high = 2,
            size = (self.n_size, 1),
            dtype = cp.uint16
        )

        # 对那些被选中的个体，随机选择一个基因位点进行操作
        col_ind = cp.random.randint(
            low = 0, 
            high = self.n_genes, 
            size = (self.n_size, 1),
            dtype = cp.uint32
        )

        # 对每个基因位点，随机选择一个 bit 进行翻转
        # 每个基因位点存储在一个 unsigned int 中，是 32 位的
        # 所以生成 [0, 31] 也就是 [0, 32) 的随机整数来决定翻转哪一个 bit
        bit_ind = cp.random.randint(
            low = 0, 
            high = 32,
            size = (self.n_size, 1),
            dtype = cp.uint32
        )

        # 设定 GPU 资源参数，开始调用
        # 原地修改 population
        BLOCK_SIZE = 16
        dim_block = (1, BLOCK_SIZE, )
        dim_grid = (1, ceil(self.n_size/dim_block[0]),)
        bit_flop_kernel(
            dim_grid,
            dim_block,
            (self.genes, self.n_size, self.n_genes, col_ind, bit_ind, mutate, )
        )

    # 产生下一代幼崽
    def born_next_generation(self) -> None:

        # 给每个 population 找伴侣进行 cross
        # 也就是生成 pop_size 个范围在 [0, pop_size-1] 的随机数
        # 0, 1, 2, ..., pop_size-1 被取到的概率就是之前算出来的 probs
        match = cp.random.choice(
            a = self.n_size,
            size = self.n_size,
            replace = True,
            p = self.scores
        ).astype(cp.uint32)

        born = cp.zeros(
            shape = (2*self.n_size, self.n_genes, ),
            dtype = cp.uint32
        )

        # 给每个 born 分配一个 thread 去做
        BLOCK_SIZE = 16
        dim_block = (2, BLOCK_SIZE, )
        dim_grid = (1, ceil(self.n_size/dim_block[1]), )
        cross_kernel(
            dim_grid,
            dim_block,
            (self.genes, match, born, self.n_size, self.n_genes, )
        )

        self.genes = cp.concatenate(
            tup = (self.genes, born,), 
            axis = 0
        )

        self.n_size = self.genes.shape[0]

    # 过滤与筛选
    def select_next_gen(self) -> None:

        # 那么就按照 probs 作为概率，选出下一代
        next_gen_indexes = cp.unique(cp.random.choice(
            self.n_size, 
            size = self.n_size, 
            p = self.scores
        ))

        self.genes = self.genes[next_gen_indexes, :]
        self.n_size = self.genes.shape[0]
