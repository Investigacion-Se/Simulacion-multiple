import numpy as np

def AvanzarSimulacion(funcion, dt, estadoActual):
    k1 = funcion(estadoActual)
    k2 = funcion(estadoActual + k1 * estadoActual / 2)
    k3 = funcion(estadoActual + k2 * estadoActual / 2)
    k4 = funcion(estadoActual + k3 * estadoActual)

    paso = np.array(k1 + 2 * k2 + 2 * k3 + k4)

    return estadoActual + (dt / 6) * paso


