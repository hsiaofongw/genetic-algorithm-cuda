from genetic import Population

# 生成初始种群，种群有 pop_size 个个体，每个个体有 problem_size 个基因位点
pop_size = 100
problem_size = 10
population = Population.generate_initial_population(
    pop_size, 
    problem_size
)

