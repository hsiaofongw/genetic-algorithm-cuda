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

    # 最大承载量
    maximum_size = 10000

    def __init__(self, genes: cp.ndarray) -> None:
        self.genes = genes
        self.n_genes = genes.shape[1]

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
    def calculate_scores(self, environment: cp.ndarray) -> None:

        # 将 genes (基因型) 转化为 phenotype (表现型)
        # routes 的每行是一个 route，一个 route 是一个关于 vertex 的列表
        routes = self.phenotype

        # distance_mat[i, j] 决定了 vertex i -> vertex j 的距离
        # 这个届时换成特定的距离矩阵
        distance_mat = environment

        # 确定要调用的 GPU 资源的规模，参看 CUDA Programming Guide
        # out_distances[i, j] 代表 routes[i] 的从 routes[i, j] 走到 routes[i, j+1] 的距离
        # 如果 j+1 == routes.shape[1]，则 out_distances[i, j] = 距离[routes[i, j], routes[i, 0]]
        BLOCK_SIZE = 16
        dim_block = (BLOCK_SIZE, BLOCK_SIZE, )
        dim_grid = (ceil(self.n_genes/dim_block[0]), ceil(self.genes.shape[0]/dim_block[1]), )
        out_distances = cp.zeros(shape = self.genes.shape, dtype = cp.float32)
        distance_kernel(
            dim_grid, 
            dim_block, 
            (routes, distance_mat, out_distances, self.n_genes, self.genes.shape[0],)
        )

        # distances_per_sample[i] 是 routes[i] 走过的总距离，
        # 也就是，从 routes[i, 0] 出发，再回到 routes[i, 0] 走过的总距离
        distances_per_sample = cp.sum(out_distances, axis=1)

        # 距离越长，得分越低
        self.scores = cp.subtract(0, distances_per_sample)
    
    # 计算每个个体的生存几率
    def calculate_survival_chance(self):

        # 首先获取适应度
        scores = self.scores

        # 升序排列，数字越高，对应的适应度越高
        order = cp.argsort(cp.argsort(scores))

        # probs[i] 是第 i 个个体被选入下一代的概率
        self.probs = cp.divide(order, cp.sum(order))


    # 通过基因变异引入新性状
    def populations_mutate(self) -> None:

        # 随机确定选哪些个 population 进行 mutate
        # 如果 mutate[i] == 1，则表示要对 self.genes[i] 进行 mutate 操作
        mutate = cp.random.randint(
            low = 0,
            high = 2,
            size = (self.genes.shape[0], 1),
            dtype = cp.uint16
        )

        # 对那些被选中的个体，随机选择一个基因位点进行操作
        col_ind = cp.random.randint(
            low = 0, 
            high = self.n_genes, 
            size = (self.genes.shape[0], 1),
            dtype = cp.uint32
        )

        # 对每个基因位点，随机选择一个 bit 进行翻转
        # 每个基因位点存储在一个 unsigned int 中，是 32 位的
        # 所以生成 [0, 31] 也就是 [0, 32) 的随机整数来决定翻转哪一个 bit
        bit_ind = cp.random.randint(
            low = 0, 
            high = 32,
            size = (self.genes.shape[0], 1),
            dtype = cp.uint32
        )

        # 设定 GPU 资源参数，开始调用
        BLOCK_SIZE = 16
        dim_block = (1, BLOCK_SIZE, )
        dim_grid = (1, ceil(self.genes.shape[0]/dim_block[1]),)
        bit_flop_kernel(
            dim_grid,
            dim_block,
            (self.genes, self.genes.shape[0], self.n_genes, col_ind, bit_ind, mutate, )
        )

    # 产生下一代幼崽
    def born_next_generation(self) -> None:

        # 给每个个体找伴侣进行 cross
        # 也就是为每个个体生成范围在 [0, pop_size-1] 的随机数
        # 0, 1, 2, ..., pop_size-1 被取到的概率就是之前算出来的 probs
        match = cp.random.choice(
            a = self.genes.shape[0],
            size = self.genes.shape[0],
            replace = True,
            p = self.probs
        ).astype(cp.uint32)

        born = cp.zeros(
            shape = (2*self.genes.shape[0], self.n_genes, ),
            dtype = cp.uint32
        )

        # 给每个 born 分配一个 thread 去做
        BLOCK_SIZE = 16
        dim_block = (2, BLOCK_SIZE, )
        dim_grid = (1, ceil(self.genes.shape[0]/dim_block[1]), )
        cross_kernel(
            dim_grid,
            dim_block,
            (self.genes, match, born, self.genes.shape[0], self.n_genes, )
        )

        self.genes = cp.concatenate(
            tup = (self.genes, born,), 
            axis = 0
        )

    # 过滤与筛选
    def select_next_gen(self) -> None:

        # 那么就按照 probs 作为概率，选出下一代
        next_gen_indexes = cp.unique(cp.random.choice(
            self.genes.shape[0], 
            size = self.genes.shape[0], 
            p = self.probs
        ))

        self.genes = self.genes[next_gen_indexes, :]
    
    # 获取最高分数
    def get_maximum_score(self) -> cp.ndarray:
        return cp.amax(self.scores)
    
    # 获取种群规模
    def get_population_size(self) -> cp.ndarray:
        return self.genes.shape[0]
    
    # 更新生存几率
    def update_chance(self, environment: cp.ndarray) -> None:
        self.calculate_phenotype()
        self.calculate_scores(environment)
        self.calculate_survival_chance()

    # 更新状态
    def evolve(self, environment: cp.ndarray) -> None:

        # 更新适应度
        self.update_chance(environment)

        # 筛选
        self.select_next_gen()

        # 更新适应度
        self.update_chance(environment)

        # 变异
        self.populations_mutate()

        # 更新适应度
        self.update_chance(environment)

        # 杂交（使变异产生的新基因型扩散）
        self.born_next_generation()

        # 更新适应度
        self.update_chance(environment)

        # 减少数量
        while self.get_population_size() >= self.maximum_size:
            self.select_next_gen()
            self.update_chance(environment)
        