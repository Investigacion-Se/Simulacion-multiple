import numpy as np

def CampanaDeGauss(x, mu, sigma):
    factor = 1 / (np.sqrt(2 * np.pi) * sigma)
    return factor * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))

def CampanaDeGaussDerivada(x, mu, sigma):
    factor = (x - mu) / (np.sqrt(2 * np.pi) * sigma ** 3)
    y = factor * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))
    return np.fft.fft(np.fft.ifftshift(y / np.sum(CampanaDeGauss(x, mu, sigma))))