import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import gamma

def sphere_function(x):
    return np.sum(x ** 2)

def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def ackley(x):
    a, b, c = 20, 0.2, 2 * np.pi
    n = len(x)
    sum_sq = np.sum(x ** 2)
    sum_cos = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + a + np.exp(1)

def fitness_function(x):
    return sphere_function(x)

def levy_flight(beta=1.5, scale=0.3):
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn() * sigma
    v = np.random.randn()
    return scale * u / abs(v) ** (1 / beta)

class BWO:
    def __init__(self, dim, initial_population=None, n_whales=30, Tmax=100, bounds=(-1, 1)):
        self.dim = dim
        self.n = n_whales
        self.Tmax = Tmax
        self.bounds = bounds
        self.population = np.random.uniform(bounds[0], bounds[1], (self.n, dim))
        self.fitness = np.array([fitness_function(ind) for ind in self.population])
        self.best = self.population[np.argmin(self.fitness)].copy()
        self.best_fitness = np.min(self.fitness)
        self.history = []

    def optimize(self):
        for T in range(self.Tmax):
            B0 = np.random.rand()
            Bf = B0 * (1 - T / (2 * self.Tmax))
            Wf = 0.1 - 0.05 * T / self.Tmax
            C2 = 2 * Wf * self.n

            for i in range(self.n):
                r = np.random.rand()
                if Bf > 0.5:  # Exploration
                    r1, r2 = np.random.rand(), np.random.rand()
                    j = np.random.randint(0, self.dim)
                    r_idx = np.random.randint(0, self.n)
                    if j % 2 == 0:
                        self.population[i, j] += (self.population[r_idx, j] - self.population[i, j]) / (1 + r1) * np.sin(2 * np.pi * r2)
                    else:
                        self.population[i, j] += (self.population[r_idx, j] - self.population[i, j]) / (1 + r1) * np.cos(2 * np.pi * r2)

                else:  # Exploitation
                    r3, r4 = np.random.rand(), np.random.rand()
                    r_idx = np.random.randint(0, self.n)
                    C1 = 2 * r4 * (1 - T / self.Tmax)
                    LF = levy_flight()
                    self.population[i] = (r3 * self.best - r4 * self.population[i] +
                                          C1 * LF * (self.population[r_idx] - self.population[i]))

                # Whale Fall
                if np.random.rand() < Wf:
                    r5, r6, r7 = np.random.rand(), np.random.rand(), np.random.rand()
                    r_idx = np.random.randint(0, self.n)
                    step = (self.bounds[1] - self.bounds[0]) * np.exp(-C2 * T / self.Tmax)
                    self.population[i] = r5 * self.population[i] - r6 * self.population[r_idx] + r7 * step

                self.population[i] = np.clip(self.population[i], *self.bounds)
                self.fitness[i] = fitness_function(self.population[i])

                if self.fitness[i] < self.best_fitness:
                    self.best = self.population[i].copy()
                    self.best_fitness = self.fitness[i]

            self.history.append(self.best_fitness)
            # if T % 10 == 0 or T == self.Tmax - 1:
            #     print(f"BWO Iter {T}: Best fitness = {self.best_fitness:.4f}")

        return self.best, self.best_fitness

class ImprovedBWO:
    def __init__(self, dim, initial_population=None, n_whales=30, Tmax=100, bounds=(-1, 1)):
        self.dim = dim
        self.n = n_whales
        self.Tmax = Tmax
        self.bounds = bounds
        self.alpha_min = 0.3
        self.alpha_max = 0.7
        self.stuck_counter = 0
        self.prev_best_fitness = float('inf')
        self.stagnation_limit = 10
        # self.population = np.random.uniform(-1.5, 1.5, (self.n, dim))
        if initial_population is not None:
            self.population = initial_population.copy()
        else:
            self.population = np.random.uniform(-1.5, 1.5, (self.n, dim))
        self.fitness = np.array([fitness_function(ind) for ind in self.population])
        self.best = self.population[np.argmin(self.fitness)]
        self.best_fitness = np.min(self.fitness)
        self.history = []

    def optimize(self):
        for T in range(self.Tmax):
            B0 = np.random.rand()
            Bf = B0 * (1 - T / (2 * self.Tmax))
            # Bf = 0.8 * (1 - T / self.Tmax) + 0.2
            Wf = 0.1 - 0.05 * T / self.Tmax
            C2 = 2 * Wf * self.n

            num_elites = max(1, int(0.3 * self.n))
            elite_indices = np.argsort(self.fitness)[:num_elites]
            elites = self.population[elite_indices]

            for i in range(self.n):
                r = np.random.rand()
                if Bf > 0.5:  # Exploration
                    r1, r2 = np.random.rand(), np.random.rand()
                    j = np.random.randint(0, self.dim)
                    r_idx = np.random.randint(0, self.n)
                    if j % 2 == 0:
                        self.population[i, j] += (self.population[r_idx, j] - self.population[i, j]) / (1 + r1) * np.sin(2 * np.pi * r2)
                    else:
                        self.population[i, j] += (self.population[r_idx, j] - self.population[i, j]) / (1 + r1) * np.cos(2 * np.pi * r2)

                else:  # Exploitation
                    f_i = self.fitness[i]
                    f_best = np.min(self.fitness)
                    f_worst = np.max(self.fitness)
                    # highest fitness is choosen for exploitation
                    # f_best = np.max(self.fitness)
                    # f_worst = np.min(self.fitness)

                    if f_best == f_worst:
                        alpha_i = self.alpha_min
                    else:
                        alpha_i = self.alpha_min + (self.alpha_max - self.alpha_min) * (
                            1 - (f_i - f_worst) / (f_best - f_worst))
                        # Scale alpha_i so individuals with higher fitness get higher alpha_i
                        # alpha_i = self.alpha_min + (self.alpha_max - self.alpha_min) * (
                        #     (f_i - f_worst) / (f_best - f_worst))

                    # Pick random elite from top-K
                    elite = elites[np.random.randint(0, len(elites))]

                    # Beluga movement toward random elite using Levy
                    r3, r4 = np.random.rand(), np.random.rand()
                    LF = levy_flight(scale=0.3)
                    self.population[i] = (r3 * elite - r4 * self.population[i] +
                                        alpha_i * LF * (elite - self.population[i]))

                # Whale Fall
                if np.random.rand() < Wf:
                    r5, r6, r7 = np.random.rand(), np.random.rand(), np.random.rand()
                    r_idx = np.random.randint(0, self.n)
                    step = (self.bounds[1] - self.bounds[0]) * np.exp(-C2 * T / self.Tmax)
                    self.population[i] = r5 * self.population[i] - r6 * self.population[r_idx] + r7 * step

                self.population[i] = np.clip(self.population[i], *self.bounds)
                self.fitness[i] = fitness_function(self.population[i])

                if self.fitness[i] < self.best_fitness:
                    self.best = self.population[i].copy()
                    self.best_fitness = self.fitness[i]
                
            if self.best_fitness >= self.prev_best_fitness - 1e-6:  # no improvement
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0  # reset on improvement
            self.prev_best_fitness = self.best_fitness

            if self.stuck_counter >= self.stagnation_limit:
                num_to_restart = int(0.3 * self.n)  # Restart worst 30%
                worst_indices = np.argsort(self.fitness)[-num_to_restart:]
                self.population[worst_indices] = np.random.uniform(
                    self.bounds[0], self.bounds[1], (num_to_restart, self.dim)
                )
                self.fitness[worst_indices] = [fitness_function(ind)
                                            for ind in self.population[worst_indices]]
                
                # Check if new whales are better
                current_best_idx = np.argmin(self.fitness)
                if self.fitness[current_best_idx] < self.best_fitness:
                    self.best_fitness = self.fitness[current_best_idx]
                    self.best = self.population[current_best_idx].copy()
                
                self.stuck_counter = 0  # reset counter
                # print(f"Re-initialized {num_to_restart} whales due to stagnation.")

            self.history.append(self.best_fitness)
            # if T % 10 == 0 or T == self.Tmax - 1:
            #     print(f"BWO Iter {T}: Best fitness = {self.best_fitness:.4f}")

        return self.best, self.best_fitness


runs = 30
dim = 30
Tmax = 100

bwo_results = []
ibwo_results = []

for r in range(runs):
    initial_pop = np.random.uniform(-1.5, 1.5, (50, dim))

    bwo = BWO(dim=dim, initial_population=initial_pop, n_whales=50, Tmax=Tmax)
    _, best_val = bwo.optimize()
    bwo_results.append(best_val)

    ibwo = ImprovedBWO(dim=dim, initial_population=initial_pop, n_whales=50, Tmax=Tmax)
    _, best_val_ibwo = ibwo.optimize()
    ibwo_results.append(best_val_ibwo)

print("\n--- Statistics over 30 runs ---")
print(f"BWO  mean: {np.mean(bwo_results):.4e}, std: {np.std(bwo_results):.4e}")
print(f"IBWO mean: {np.mean(ibwo_results):.4e}, std: {np.std(ibwo_results):.4e}")
