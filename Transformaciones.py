import numpy as np

def TransformacionDelCampo(campoActual, cantidadPuntos, kernels, coeficientes, cantidadCoeficientes):
    campoTransformado = np.fft.fft(campoActual)
    nuevosCoeficientes = np.zeros([cantidadPuntos, cantidadCoeficientes], dtype = complex)

    for i in range(cantidadPuntos):
        matriz = np.zeros([cantidadCoeficientes, cantidadCoeficientes])
        for j in range(len(coeficientes)):
            matriz += coeficientes[j] * kernels[j, i]
        
        values, vectors = np.linalg.eig(matriz)
        nuevosCoeficientes[i, :] = values
        campoTransformado[:, i] = np.linalg.inv(vectors) @ campoTransformado[:, i]

    return campoTransformado, nuevosCoeficientes
    

def DestransformacionDelCampo(campoTransformado, cantidadPuntos, kernels, coeficientes, cantidadCoeficientes):
    
    for i in range(cantidadPuntos):
        matriz = np.zeros([cantidadCoeficientes, cantidadCoeficientes])
        for j in range(len(coeficientes)):
            matriz += coeficientes[j] * kernels[j, i] 
        
        _, vectors = np.linalg.eig(matriz)
        campoTransformado[:, i] = vectors @ campoTransformado[:, i]

    campoDestrasformado = np.fft.ifft(campoTransformado)

    return np.real(campoDestrasformado)