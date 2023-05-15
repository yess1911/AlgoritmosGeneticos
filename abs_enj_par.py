import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pyswarms.single.global_best import GlobalBestPSO

# Define la función de prueba
def absolute(x):
    result = np.sum(np.abs(x))
    performance.append(result)
    return result

# Define los límites para la búsqueda de la solución óptima
bounds = (np.array([-5, -5]), np.array([5, 5]))

# Define una función para guardar los valores de la función de desempeño en cada iteración
def callback(positions, convergence):
    performance.append(convergence)

# Configura las opciones de la optimización
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# Optimiza la función utilizando PSO
performance = []
optimizer = GlobalBestPSO(n_particles=20, dimensions=2, options=options, bounds=bounds, ftol=-1, ftol_iter=50)
best_pos, best_cost = optimizer.optimize(absolute, iters=50, verbose=False)

# Imprime el valor óptimo de la función y la mejor solución encontrada
print("Valor óptimo de la función:", best_cost)
print("Mejor solución encontrada:", best_pos)

# Genera una malla de puntos en 3D
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = absolute([X[i, j], Y[i, j]])

# Grafica la función en 3D
fig = plt.figure("Gráfica función Absoluto")
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Grafica la evolución de la función de desempeño
plt.figure("Gráfica evolución de desempeño")
plt.plot(performance)
plt.xlabel('Iteración')
plt.ylabel('Valor de la función de desempeño')
plt.show()

# Graficar valor óptimo de la función y mejor solución encontrada
fig, ax = plt.subplots()
ax.plot(performance)
ax.set_xlabel('Iteración')
ax.set_ylabel('Valor de la función de desempeño')
ax2 = ax.twinx()
ax2.plot(np.full(len(performance), best_cost[0]), 'r--')
ax2.plot([np.argmin(performance)], [best_cost], 'go')
ax2.set_ylabel('Valor óptimo de la función')
plt.show()

