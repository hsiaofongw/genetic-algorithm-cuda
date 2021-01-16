import cupy as cp
from genetic import Population

# 生成初始种群，种群有 pop_size 个个体，每个个体有 problem_size 个基因位点
pop_size = 100
problem_size = 12
population = Population.generate_initial_population(
    pop_size, 
    problem_size
)

distance_mat = cp.random.randint(
    low = 1,
    high = 1000,
    size = (problem_size, problem_size,)
).astype(cp.float32)

max_evolves_n = 100
for i in range(max_evolves_n):
    population.evolve(distance_mat)
    population.update_chance(distance_mat)
    print("At generation: %s\n Score: %s\n" % (str(i), population.get_maximum_score(),))