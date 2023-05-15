import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Define la función de prueba
def absolute(x):
    return np.sum(np.abs(x))

# Define los límites para la búsqueda de la solución óptima
bounds = [(-5, 5), (-5, 5)]

# Define los parámetros del algoritmo de colonia de hormigas
num_ants = 10
num_iterations = 200
alpha = 1
beta = 1
rho = 0.1
q0 = 0.9

# Inicializa la matriz de feromonas
pheromone = np.ones((len(bounds), num_ants))

# Optimiza la función utilizando el algoritmo de colonia de hormigas
best_solution = None
best_fitness = np.inf

# Inicializa las variables para registrar la evolución de la función de desempeño y la concentración de feromonas
performance = []
pheromone_history = []
for iteration in range(num_iterations):
    solutions = np.zeros((num_ants, len(bounds)))
    fitnesses = np.zeros(num_ants)

    # Construye soluciones para cada hormiga
    for ant in range(num_ants):
        current_node = np.random.randint(len(bounds))
        unvisited = list(range(len(bounds)))
        unvisited.remove(current_node)
        solutions[ant, current_node] = np.random.uniform(*bounds[current_node])
        
        for i in range(1, len(bounds)):
            probabilities = []
            for node in unvisited:
                numerator = pheromone[current_node, node]**alpha * (1/np.abs(solutions[ant, current_node] - bounds[node][0])**beta)
                denominator = np.sum([pheromone[current_node, j]**alpha * (1/np.abs(solutions[ant, current_node] - bounds[j][0])**beta) for j in unvisited])
                probabilities.append(numerator/denominator)
            probabilities = np.array(probabilities)
            
            if np.random.uniform() < q0:
                next_node = unvisited[np.argmax(probabilities)]
            else:
                next_node = np.random.choice(unvisited, p=probabilities/probabilities.sum())
            
            solutions[ant, next_node] = np.random.uniform(*bounds[next_node])
            current_node = next_node
            unvisited.remove(current_node)
        
        fitnesses[ant] = absolute(solutions[ant])

    # Actualiza la matriz de feromonas
    delta_pheromone = np.zeros((len(bounds), num_ants))
    for ant in range(num_ants):
        for i in range(len(bounds)):
            delta_pheromone[i, ant] = 1/fitnesses[ant] if solutions[ant, i] >= 0 else -1/fitnesses[ant]
    pheromone = (1-rho)*pheromone + rho*delta_pheromone

    # Actualiza la mejor solución encontrada
    iteration_best_idx = np.argmin(fitnesses)
    if fitnesses[iteration_best_idx] < best_fitness:
        best_solution = solutions[iteration_best_idx]
        best_fitness = fitnesses[iteration_best_idx]

        # Registra el mejor rendimiento de la iteración
        performance.append(best_fitness)
            
        # Registra la concentración de feromonas
        pheromone_history.append(pheromone.copy())

    # Imprime información de la iteración
    #print(f"Iteration {iteration+1}/{num_iterations}: best fitness = {best_fitness:.6f}")

# Imprime el valor óptimo de la función y la mejor solución encontrada
print("Valor óptimo de la función:", best_fitness)
print("Mejor solución encontrada:", best_solution)

#Grafica la evolución de la función de desempeño
plt.figure("función de desempeño")
plt.plot(performance)
plt.xlabel('Iteración')
plt.ylabel('Valor de la función de desempeño')
plt.title('Evolución de la función de desempeño')

#Grafica la evolución de la concentración de feromonas
plt.figure("Evolución de la concentración de feromonas")
for i in range(len(bounds)):
    plt.plot(np.array(pheromone_history)[:,i,:].flatten(), label=f'Feromonas {i}')
plt.xlabel('Iteración')
plt.ylabel('Concentración de feromonas')
plt.title('Evolución de la concentración de feromonas')
plt.legend()

#Grafica la mejor solución encontrada
plt.figure("Mejor solución")
plt.plot(performance)
plt.xlabel('Iteración')
plt.ylabel('Valor de la función de desempeño')
plt.title('Mejor solución encontrada')
for i in range(1, len(best_solution)):
    plt.plot([i-1, i], [absolute(best_solution[i-1]), absolute(best_solution[i])], 'k-', lw=1)
    plt.show()


#Grafica 3D de la función del valor absoluto
def f(x, y):
    return np.abs(x) + np.abs(y)

fig = plt.figure("Absoluto en 3D")
ax = fig.add_subplot(111, projection='3d')
x = y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
ax.plot_surface(X, Y, Z)
plt.show()