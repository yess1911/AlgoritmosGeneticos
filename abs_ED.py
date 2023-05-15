import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

#Parametros para graficar
def objective_function(x):
    return np.abs(x[0]) + np.abs(x[1])

def evaluate(population):
    return np.array([objective_function(ind) for ind in population])


def create_children(population, F, CR):
    children = []
    for i, parent in enumerate(population):
        a, b, c = population[np.random.choice(len(population), 3, replace=False)]
        mutant = a + F * (b - c)
        trial = np.where(np.random.rand(len(parent)) < CR, mutant, parent)
        children.append(trial)
    return np.array(children)

def benchmark_function(x):
    return -np.abs(x[0]) - np.abs(x[1])

def de_optimization(bounds, pop_size=100, num_generations=100, f=0.5, cr=0.5):
    # Inicializar la población
    pop = np.random.uniform(bounds[:, 0], bounds[:, 1], (pop_size, bounds.shape[0]))
    # Evaluar la población inicial
    fitness = np.array([benchmark_function(p) for p in pop])
    # Guardar la mejor solución encontrada y su fitness
    best_sol = pop[np.argmin(fitness)]
    best_fit = np.min(fitness)
    # Guardar el fitness de la mejor solución en cada generación
    history = [best_fit]

    # Ciclo de evolución diferencial
    for i in range(num_generations):
        # Seleccionar 3 individuos aleatorios diferentes
        r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
        # Calcular la diferencia entre r2 y r3
        diff = pop[r2] - pop[r3]
        # Generar una nueva población mutada
        mut_pop = pop + f * diff
        # Asegurarse de que la mutación esté dentro de los límites de búsqueda
        mut_pop = np.clip(mut_pop, bounds[:, 0], bounds[:, 1])
        # Elegir aleatoriamente si cada dimensión de cada individuo de la población mutada se mantiene o se reemplaza por la correspondiente dimensión del individuo original
        mask = np.random.rand(pop_size, bounds.shape[0]) <= cr
        trial_pop = np.where(mask, mut_pop, pop)
        # Evaluar la población mutada
        trial_fitness = np.array([benchmark_function(p) for p in trial_pop])
        # Reemplazar la población anterior con la población mutada si la mutada es mejor
        replace_mask = trial_fitness < fitness
        pop = np.where(replace_mask[:, np.newaxis], trial_pop, pop)
        fitness = np.where(replace_mask, trial_fitness, fitness)
        # Actualizar la mejor solución encontrada
        current_best_fit = np.min(fitness)
        if current_best_fit < best_fit:
            best_fit = current_best_fit
            best_sol = pop[np.argmin(fitness)]
        # Guardar el fitness de la mejor solución en cada generación
        history.append(best_fit)

    return best_sol, best_fit, history

# Definir los límites de búsqueda
bounds = np.array([[-5, 5], [-5, 5]])
# Ejecutar la optimización por evolución diferencial
best_sol, best_fit, history = de_optimization(bounds)
# Imprimir los resultados
print("Mejor solución encontrada:", best_sol)
print("Función de benchmark en la mejor solución:", -best_fit)


# Definir los límites del espacio de búsqueda
bounds = np.array([[-5, 5], [-5, 5]])

# Definir los parámetros del algoritmo
population_size = 50
num_generations = 100
F = 0.5
CR = 0.7

# Generar una población inicial aleatoria
population = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(population_size, bounds.shape[0]))

# Evaluar la población inicial
fitness = evaluate(population)
best_index = np.argmin(fitness)
best_fitness = fitness[best_index]
best_solution = population[best_index]

# Graficar la función
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.abs(X) + np.abs(Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()

# Graficar la función de desempeño
history = [best_fitness]
for i in range(num_generations):
    children = create_children(population, F, CR)
    children_fitness = evaluate(children)
    for j in range(population_size):
        if children_fitness[j] < fitness[j]:
            population[j] = children[j]
            fitness[j] = children_fitness[j]
    best_index = np.argmin(fitness)
    best_fitness = fitness[best_index]
    best_solution = population[best_index]
    history.append(best_fitness)

fig, ax = plt.subplots()
ax.plot(history)
ax.set_xlabel('Generation')
ax.set_ylabel('Best fitness')
plt.show()


