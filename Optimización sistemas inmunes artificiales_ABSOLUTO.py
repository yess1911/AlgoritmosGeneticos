import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def benchmark_function(x, y):
    return np.abs(x) + np.abs(y)

def generate_random_population(size, bounds):
    return np.random.uniform(bounds[:, 0], bounds[:, 1], (size, bounds.shape[0]))

def affinity(population):
    x, y = population[:, 0], population[:, 1]
    return benchmark_function(x, y)

def select_best_antibodies(population, affinities, top_k):
    return population[np.argsort(affinities)[:top_k]]

def clone_and_mutate(antibodies, mutation_rate, bounds):
    clones = antibodies.copy()
    mutations = mutation_rate * np.random.randn(*antibodies.shape)
    mutated = antibodies + mutations
    mutated = np.clip(mutated, bounds[:, 0], bounds[:, 1])
    return mutated

def ais_optimization(bounds, pop_size=100, num_generations=100, top_k=10, mutation_rate=0.1):
    population = generate_random_population(pop_size, bounds)
    best_affinity = np.inf
    best_antibody = None
    performance_history = []

    for _ in range(num_generations):
        affinities = affinity(population)
        top_antibodies = select_best_antibodies(population, affinities, top_k)
        clones = clone_and_mutate(top_antibodies, mutation_rate, bounds)
        population = np.vstack((population, clones))

        min_affinity = np.min(affinities)
        if min_affinity < best_affinity:
            best_affinity = min_affinity
            best_antibody = population[np.argmin(affinities)]
        
        performance_history.append(best_affinity)
    return best_antibody, best_affinity, performance_history

# Graficar la función en 3D
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = benchmark_function(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()

# Ejecutar la optimización AIS
bounds = np.array([[-5, 5], [-5, 5]])
best_antibody, best_affinity, performance_history = ais_optimization(bounds)

# Graficar la función de desempeño
fig, ax = plt.subplots()
ax.set_title("Función de desempeño (Afinidad)")
ax.set_xlabel("Generaciones")
ax.set_ylabel("Afinidad")
ax.plot(performance_history, 'r-')

plt.show()

print("Mejor solución encontrada:", best_antibody)
print("Función de benchmark en la mejor solución:", best_affinity)
