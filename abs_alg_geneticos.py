import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import differential_evolution

# Define la función de prueba
def absolute(x):
    return np.sum(np.abs(x))

# Define los límites para la búsqueda de la solución óptima
bounds = [(-5, 5), (-5, 5)]

# Define una función para guardar los valores de la función de desempeño en cada iteración
def callback(xk, convergence):
    performance.append(convergence)

# Optimiza la función utilizando el algoritmo genético
performance = []
result = differential_evolution(absolute, bounds, callback=callback)

# Imprime el valor óptimo de la función y la mejor solución encontrada
print("Valor óptimo de la función:", result.fun)
print("Mejor solución encontrada:", result.x)

# Genera una malla de puntos en 3D
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = absolute([X[i, j], Y[i, j]])

# Grafica la función en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Grafica la evolución de la función de desempeño
plt.figure()
plt.plot(performance)
plt.xlabel('Iteración')
plt.ylabel('Valor de la función de desempeño')
plt.show()