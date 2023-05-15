import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Definir la función objetivo
def objective_function(x):
    return np.abs(x[0]) + np.abs(x[1])

# Definir la función de evaluación
def evaluate(population):
    return np.array([objective_function(ind) for ind in population])

# Definir la función de exploración local
def local_search(solution, bounds):
    neighbor = solution + np.random.uniform(-1, 1, size=solution.shape)
    neighbor = np.clip(neighbor, bounds[:, 0], bounds[:, 1])
    return neighbor

# Definir la función de optimización por colonia de abejas
def bee_optimization(bounds, num_employed=20, num_onlookers=20, num_cycles=100):
    num_dimensions = bounds.shape[0]
    
    # Inicializar las abejas empleadas
    employed = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(num_employed, num_dimensions))
    employed_fitness = evaluate(employed)
    
    # Inicializar las abejas observadoras
    onlookers = np.zeros_like(employed)
    onlookers_fitness = np.zeros(num_onlookers)
    
    # Guardar la mejor solución encontrada y su fitness
    best_sol = employed[np.argmin(employed_fitness)]
    best_fit = np.min(employed_fitness)
    
    # Guardar el fitness de la mejor solución en cada ciclo
    history = [best_fit]
    
    # Ciclo de optimización por colonia de abejas
    for cycle in range(num_cycles):
        # Fase de exploración de las abejas empleadas
        for i in range(num_employed):
            neighbor = local_search(employed[i], bounds)
            neighbor_fitness = objective_function(neighbor)
            if neighbor_fitness < employed_fitness[i]:
                employed[i] = neighbor
                employed_fitness[i] = neighbor_fitness
        
        # Calcular la probabilidad de selección de las abejas observadoras
        total_fitness = np.sum(employed_fitness)
        selection_probs = employed_fitness / total_fitness
        
        # Fase de selección de las abejas observadoras
        for j in range(num_onlookers):
            selected_bee = np.random.choice(num_employed, p=selection_probs)
            onlookers[j] = local_search(employed[selected_bee], bounds)
            onlookers_fitness[j] = objective_function(onlookers[j])
        
        # Fase de actualización de las abejas empleadas
        for i in range(num_employed):
            if onlookers_fitness[i] < employed_fitness[i]:
                employed[i] = onlookers[i]
                employed_fitness[i] = onlookers_fitness[i]
        
        # Actualizar la mejor solución encontrada
        current_best_fit = np.min(employed_fitness)
        if current_best_fit < best_fit:
            best_fit = current_best_fit
            best_sol = employed[np.argmin(employed_fitness)]
        
        # Guardar el fitness de la mejor solución en cada ciclo
        history.append(best_fit)
    
    return best_sol, best_fit, history

# Definir los límites del espacio de búsqueda
bounds = np.array([[-5, 5], [-5, 5]])

# Definir los parámetros del algoritmo
num_employed = 30
num_onlookers = 30
num_cycles = 100

#Ejecutar la optimización por colonia de abejas
best_sol, best_fit, history = bee_optimization(bounds, num_employed, num_onlookers, num_cycles)

#Imprimir los resultados
print("Mejor solución encontrada:", best_sol)
print("Función objetivo en la mejor solución:", best_fit)

#Graficar la función objetivo en un espacio tridimensional
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

#Graficar la evolución del mejor fitness en cada ciclo
fig, ax = plt.subplots()
ax.plot(history)
ax.set_xlabel('Cycle')
ax.set_ylabel('Best fitness')
plt.show()
