import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math
from scipy.special import gamma

# Load and preprocess the dataset
data_path = 'kddcup.data_10_percent.csv'
column_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", 
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", 
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", 
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", 
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", 
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", 
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", 
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", 
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack"
]

df = pd.read_csv(data_path, header=None, names=column_names)
df = df[df['attack'].isin(['normal.', 'neptune.', 'smurf.'])]
df_sample = df.sample(n=200, random_state=42).reset_index(drop=True)

selected_indices = [2, 3, 4, 5, 6, 12, 23, 25, 26, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
selected_features = [column_names[i] for i in selected_indices]
X = df_sample[selected_features]

categorical_cols = ['service', 'flag']
X = pd.get_dummies(X, columns=categorical_cols)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = df_sample['attack'].apply(lambda x: 0 if x == 'normal.' else 1).values

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Define the neural network model
class SimpleANN:
    def __init__(self, n_input, n_hidden, n_output):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    def set_weights(self, weight_vector):
        idx = 0
        size_W1 = self.n_input * self.n_hidden
        self.W1 = weight_vector[idx: idx + size_W1].reshape(self.n_input, self.n_hidden)
        idx += size_W1

        self.b1 = weight_vector[idx: idx + self.n_hidden].reshape(1, self.n_hidden)
        idx += self.n_hidden

        size_W2 = self.n_hidden * self.n_output
        self.W2 = weight_vector[idx: idx + size_W2].reshape(self.n_hidden, self.n_output)
        idx += size_W2

        self.b2 = weight_vector[idx: idx + self.n_output].reshape(1, self.n_output)

    def forward(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        output = self.sigmoid(z2)
        return output

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

def get_num_params(n_input, n_hidden, n_output):
    return n_input * n_hidden + n_hidden + n_hidden * n_output + n_output

# def fitness_function(weight_vector, model, X, y):
#     model.set_weights(weight_vector)
#     y_pred = model.forward(X)
#     mse = np.mean((y.reshape(-1, 1) - y_pred) ** 2)
#     return mse

def fitness_function(weight_vector, model, X_train, y_train, X_val, y_val,
                     alpha=1.0, beta=0.0001, gamma=0.05):
    # Set weights
    model.set_weights(weight_vector)

    # Forward passes
    y_train_pred = model.forward(X_train)
    y_val_pred = model.forward(X_val)

    # Compute errors
    mse_val = np.mean((y_val.reshape(-1, 1) - y_val_pred) ** 2)
    mse_train = np.mean((y_train.reshape(-1, 1) - y_train_pred) ** 2)
    generalization_gap = abs(mse_val - mse_train)

    # Compute complexity penalty for 1 hidden layer
    n_in = model.n_input
    n_hidden = model.n_hidden
    n_out = model.n_output
    complexity_penalty = (n_in * n_hidden + n_hidden) + (n_hidden * n_out + n_out)

    # Weighted sum fitness
    fitness = alpha * mse_val + beta * complexity_penalty + gamma * generalization_gap
    return fitness


class MMACOArchitectureSearch:
    def __init__(self, X, y, n_input, n_output, hidden_range=(2, 15), n_ants=10, n_iterations=10,
                 pheromone_min=1.0, pheromone_max=9.0, evaporation_rate=0.1):
        self.X = X
        self.y = y
        self.n_input = n_input
        self.n_output = n_output
        self.hidden_range = hidden_range
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        
        # MMAS specific parameters
        self.pheromone_min = pheromone_min
        self.pheromone_max = pheromone_max
        self.evaporation_rate = evaporation_rate
        
        # Initialize pheromone trails to max value
        n_options = hidden_range[1] - hidden_range[0] + 1
        self.pheromone = np.full(n_options, pheromone_max)
        
        # For storing best solution
        self.best_hidden = None
        self.best_fitness = float('inf')
        self.best_iteration = 0

    def run(self):
        for iteration in range(self.n_iterations):
            hidden_choices = np.arange(self.hidden_range[0], self.hidden_range[1] + 1)
            
            # Selection probabilities with normalization
            probabilities = self.pheromone / self.pheromone.sum()
            
            # Initialization
            ants_hidden = np.random.choice(hidden_choices, size=self.n_ants, p=probabilities)
            
            fitnesses = []
            current_iteration_best_fitness = float('inf')
            current_iteration_best_hidden = None
            
            for hidden in ants_hidden:
                model = SimpleANN(self.n_input, hidden, self.n_output)
                dim = get_num_params(self.n_input, hidden, self.n_output)
                weights = np.random.uniform(-1, 1, dim)
                X_subtrain, X_val, y_subtrain, y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
                fitness = fitness_function(weights, model, X_subtrain, y_subtrain, X_val, y_val)
                fitnesses.append(fitness)
                
                # Track iteration best
                if fitness < current_iteration_best_fitness:
                    current_iteration_best_fitness = fitness
                    current_iteration_best_hidden = hidden
                
                # Track global best
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_hidden = hidden
                    self.best_iteration = iteration

            # Pheromone evaporation for all trails
            self.pheromone *= (1 - self.evaporation_rate)
            
            # Apply pheromone bounds after evaporation
            self.pheromone = np.clip(self.pheromone, self.pheromone_min, self.pheromone_max)
            
            # Update pheromone only for the best solution
            if current_iteration_best_hidden is not None:
                idx = current_iteration_best_hidden - self.hidden_range[0]
                delta_pheromone = 1.0 / (1.0 + current_iteration_best_fitness)
                self.pheromone[idx] += delta_pheromone
            
            # Apply pheromone bounds after update
            self.pheromone = np.clip(self.pheromone, self.pheromone_min, self.pheromone_max)
            
            print(f"Iteration {iteration+1}/{self.n_iterations}, "
                  f"Best Hidden layer: {self.best_hidden}, Best MSE: {self.best_fitness:.4f}")

        return self.best_hidden


def levy_flight(beta=1.5, scale=0.3):
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.randn() * sigma
    v = np.random.randn()
    return scale * u / abs(v) ** (1 / beta)

class BWO:
    def __init__(self, model, X, y, dim, X_val, y_val, initial_population=None, n_whales=30, Tmax=100, bounds=(-1, 1)):
        self.model = model
        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_val
        self.dim = dim
        self.n = n_whales
        self.Tmax = Tmax
        self.bounds = bounds
        self.prev_best_fitness = float('inf')
        # self.population = np.random.uniform(bounds[0], bounds[1], (self.n, dim))
        if initial_population is not None:
            self.population = initial_population.copy()
        else:
            self.population = np.random.uniform(-1.5, 1.5, (self.n, dim))
        self.fitness = np.array([fitness_function(ind, model, self.X, self.y, self.X_val, self.y_val) for ind in self.population])
        self.best = self.population[np.argmin(self.fitness)]
        self.best_fitness = np.min(self.fitness)
        self.history = []

    def optimize(self):
        for T in range(self.Tmax):
            B0 = np.random.rand()
            Bf = B0 * (1 - T / (2 * self.Tmax))
            Wf = 0.1 - 0.05 * T / self.Tmax
            C2 = 2 * Wf * self.n

            for i in range(self.n):
                # r = np.random.rand()
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
                self.fitness[i] = fitness_function(self.population[i], self.model, self.X, self.y, self.X_val, self.y_val)

                if self.fitness[i] < self.best_fitness:
                    self.best = self.population[i].copy()
                    self.best_fitness = self.fitness[i]

            self.history.append(self.best_fitness)
            # if T % 10 == 0 or T == self.Tmax - 1:
            #     print(f"BWO Iter {T}: Best MSE = {self.best_fitness:.4f}")

        return self.best, self.best_fitness
    
class ImprovedBWO:
    def __init__(self, model, X, y, dim, X_val, y_val, initial_population=None, n_whales=30, Tmax=100, bounds=(-1, 1)):
        self.model = model
        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_val
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
        self.fitness = np.array([fitness_function(ind, model, self.X, self.y, self.X_val, self.y_val) for ind in self.population])
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
                    # else:
                    #     alpha_i = self.alpha_min + (self.alpha_max - self.alpha_min) * (
                    #         (f_i - f_worst) / (f_best - f_worst))

                    # Pick random elite from top-K
                    elite = elites[np.random.randint(0, len(elites))]
                    # alpha_i = 0.8

                    # whales movement toward random elite using Levy
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
                self.fitness[i] = fitness_function(self.population[i], self.model, self.X, self.y, self.X_val, self.y_val)

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
                self.fitness[worst_indices] = [fitness_function(ind, self.model, self.X, self.y, self.X_val, self.y_val)
                                            for ind in self.population[worst_indices]]
                
                # Check if new whales are better
                current_best_idx = np.argmin(self.fitness)
                if self.fitness[current_best_idx] < self.best_fitness:
                    self.best_fitness = self.fitness[current_best_idx]
                    self.best = self.population[current_best_idx].copy()
                
                self.stuck_counter = 0  # reset counter
                print(f"Re-initialized {num_to_restart} whales due to stagnation.")

            self.history.append(self.best_fitness)
            # if T % 10 == 0 or T == self.Tmax - 1:
            #     print(f"BWO Iter {T}: Best MSE = {self.best_fitness:.4f}")

        return self.best, self.best_fitness

runs = 30
mse_bwo_list = []
acc_bwo_list = []
mse_improved_list = []
acc_improved_list = []

for run in range(runs):
    print(f"\n================ RUN {run + 1} / {runs} ================\n")
    n_input = X_train.shape[1]
    n_output = 1

    # Step 1: ACO to get optimal number of hidden layers
    aco_search = MMACOArchitectureSearch(X_train, y_train, n_input, n_output, hidden_range=(2, 15), n_ants=10, n_iterations=10)
    best_hidden = aco_search.run()

    # best_hidden = 10 # for testing (comparing BWO and IBWO with fix hidden layers)

    # Step 2: create model and get dimension
    model = SimpleANN(n_input, best_hidden, n_output)
    dim = get_num_params(n_input, best_hidden, n_output)

    initial_pop = np.random.uniform(-1.5, 1.5, (100, dim))
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Step 3: run BWO to optimize weights
    bwo = BWO(model, X_subtrain, y_subtrain, dim, X_val=X_val, y_val=y_val,
              initial_population=initial_pop, n_whales=30, Tmax=200)
    best_weights_bwo, best_mse_bwo = bwo.optimize()

    model.set_weights(best_weights_bwo)
    y_pred_test_bwo = model.forward(X_test)
    y_class_test_bwo = (y_pred_test_bwo > 0.5).astype(int)
    acc_bwo = np.mean(y_class_test_bwo.flatten() == y_test)

    mse_bwo_list.append(best_mse_bwo)
    acc_bwo_list.append(acc_bwo * 100)

    # Step 4: run ImpovedBWO to optimize weights
    bwo = ImprovedBWO(model, X_subtrain, y_subtrain, dim, X_val=X_val, y_val=y_val,
              initial_population=initial_pop, n_whales=30, Tmax=200)
    best_weights_improved_bwo, best_mse_improved_bwo = bwo.optimize()

    model.set_weights(best_weights_improved_bwo)
    y_pred_test_improved = model.forward(X_test)
    y_class_test_improved = (y_pred_test_improved > 0.5).astype(int)
    acc_improved = np.mean(y_class_test_improved.flatten() == y_test)

    mse_improved_list.append(best_mse_improved_bwo)
    acc_improved_list.append(acc_improved * 100)

methods = ['BWO', 'ImprovedBWO']

mse_means = [np.mean(mse_bwo_list), np.mean(mse_improved_list)]
mse_stds = [np.std(mse_bwo_list), np.std(mse_improved_list)]

acc_means = [np.mean(acc_bwo_list), np.mean(acc_improved_list)]
acc_stds = [np.std(acc_bwo_list), np.std(acc_improved_list)]

x = np.arange(len(methods))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 6))

# MSE bar plot (left axis)
rects1 = ax1.bar(x - width/2, mse_means, width, yerr=mse_stds, capsize=5, label='Avg MSE', color='skyblue')
ax1.set_ylabel('Average MSE')
ax1.set_title('BWO vs ImprovedBWO (30 Runs)')
ax1.set_xticks(x)
ax1.set_xticklabels(methods)

# Accuracy bar plot (right axis)
ax2 = ax1.twinx()
rects2 = ax2.bar(x + width/2, acc_means, width, yerr=acc_stds, capsize=5, label='Avg Accuracy (%)', color='lightgreen')
ax2.set_ylabel('Average Accuracy (%)')

# Combined legend
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)

# Annotate bars
def autolabel(rects, ax, fmt="{:.2f}"):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(fmt.format(height),
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1, ax1)
autolabel(rects2, ax2)

plt.tight_layout()
plt.show()

