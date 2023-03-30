import numpy as np
from numpy.fft import fft, ifftshift

def AproximacionTradicional(x, dx, orden = 1):
    largo = len(x)
    puntoMedio = int(np.floor(largo / 2))

    y = np.zeros(largo)
    valores = np.array([210, -120, 45, -10, 1])
    sumaTotal = np.sum(valores)

    for i, valor in enumerate(valores):
        valor /= 2 * (i + 1) * sumaTotal
        y[puntoMedio + i + 1] = -valor
        y[puntoMedio - i - 1] = valor

    y = fft(ifftshift(y / dx))
    return (y) ** orden

def AproximacionSegundaTradicional(x, dx):
    largo = len(x)
    puntoMedio = int(np.floor(largo / 2))

    y = np.zeros(largo)
    
    y[puntoMedio + 3] = 1 / 90
    y[puntoMedio + 2] = -3 / 20
    y[puntoMedio + 1] = 3 / 2
    y[puntoMedio + 0] = -49 / 18
    y[puntoMedio - 1] = 3 / 2
    y[puntoMedio - 2] = -3 / 20
    y[puntoMedio - 3] = 1 / 90    

    return np.fft.fft(np.fft.ifftshift(y / dx ** 2))