import numpy as np
import scipy.linalg as spl


def logL_GP(y_res, K):

    K_diag = np.diagonal(K)
    gof = np.sum(y_res * y_res / K_diag)
    logdet = np.sum(np.log(K_diag))

    logL = -0.5 * (gof + logdet + len(y_res) * np.log(2 * np.pi))

    return logL
