# CuPy 允许我们通过 Python 调用 CUDA
import cupy as cp
from math import log, ceil

from genetic import Population

# 生成初始种群，种群有 pop_size 个个体，每个个体有 problem_size 个基因位点
pop_size = 100
problem_size = 10
population = Population.generate_initial_population(
    pop_size, 
    problem_size
)

# 计算得分
population.calculate_phenotype()
population.calculate_scores()

# 过滤筛选
population.select_next_gen()

# 基因变异
population.populations_mutate()

# 生出幼崽
population.born_next_generation()

# 计算得分
population.calculate_phenotype()
population.calculate_scores()
