import numpy as np

from .numerical_characteristics import (
    fechner,
    kendall,
    kruskal,
    kurtosis,
    partial,
    pcc,
    pearson,
    sign_similarity,
    spearman,
)


def _calc_pearson_stat(
    corr: np.ndarray, threshold: float, kurt: float, n: int
) -> np.ndarray:
    def z_transform(y):
        return 0.5 * np.log((1 + y) / (1 - y))

    def statistics(y):
        if y == 1:
            return np.inf
        if y == -1:
            return -np.inf
        return np.sqrt(n / (1 + kurt)) * (z_transform(y) - z_transform(threshold))

    transformer = np.vectorize(statistics)
    return transformer(corr)


def pearson_statistics(x: np.ndarray, model: str, threshold: float) -> np.ndarray:
    """
    Calculates a matrix of statistics
    where each statistic has a standard Gaussian distribution N(0,1)
    under the assumption that the Pearson correlation
    between the i and j component of the random vector is equal to the threshold.

    Parameters
    ----------
    x : (n,N) array_like
        Sample of the size n from distribution of the N-dimensional random vector.

    model : {"gaussian", "elliptical"}
        The model according to which the random vector is distributed.

    threshold : float
        The threshold in the interval (-1, 1).

    Returns
    -------
    statistics : (N,N) ndarray
        Matrix of statistics.

    """
    n, N = x.shape
    if model == "gaussian":
        return _calc_pearson_stat(pearson(x), threshold, 0, n)
    if model == "elliptical":
        kurt = kurtosis(x)
        return _calc_pearson_stat(pearson(x), threshold, kurt, n)


def sign_similarity_statistics(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Calculates a matrix of statistics
    where each statistic has a standard Gaussian distribution N(0,1)
    under the assumption that the sign measure of similarity
    between the i and j component of the elliptical random vector
    is equal to the threshold.

    Parameters
    ----------
    x : (n,N) array_like
        Sample of the size n from distribution of the N-dimensional elliptical random vector.

    threshold : float
        The threshold in the interval (0, 1).

    Returns
    -------
    statistics : (N,N) ndarray
        Matrix of statistics.

    """
    n, N = x.shape
    return np.sqrt(n) * (sign_similarity(x) - threshold) / np.sqrt(threshold * (1 - threshold))


def fechner_statistics(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Calculates a matrix of statistics
    where each statistic has a standard Gaussian distribution N(0,1)
    under the assumption that the Fechner correlation
    between the i and j component of the elliptical random vector
    is equal to the threshold.

    Parameters
    ----------
    x : (n,N) array_like
        Sample of the size n from distribution of the N-dimensional elliptical random vector.

    threshold : float
        The threshold in the interval (-1, 1).

    Returns
    -------
    statistics : (N,N) ndarray
        Matrix of statistics.

    """
    n, N = x.shape
    return np.sqrt(n) * (fechner(x) - threshold) / np.sqrt(1 - threshold**2)


def kruskal_statistics(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Calculates a matrix of statistics
    where each statistic has a standard Gaussian distribution N(0,1)
    under the assumption that the Kruskal correlation
    between the i and j component of the elliptical random vector
    is equal to the threshold.

    Parameters
    ----------
    x : (n,N) array_like
        Sample of the size n from distribution of the N-dimensional elliptical random vector.

    threshold : float
        The threshold in the interval (-1, 1).

    Returns
    -------
    statistics : (N,N) ndarray
        Matrix of statistics.

    """
    n, N = x.shape
    return np.sqrt(n) * (kruskal(x) - threshold) / np.sqrt(1 - threshold**2)


def kendall_statistics(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Calculates a matrix of statistics
    where each statistic has a standard Gaussian distribution N(0,1)
    under the assumption that the Kendall correlation
    between the i and j component of the elliptical random vector
    is equal to the threshold.

    Parameters
    ----------
    x : (n,N) array_like
        Sample of the size n from distribution of the N-dimensional elliptical random vector.

    threshold : float
        The threshold in the interval (-1, 1).

    Returns
    -------
    statistics : (N,N) ndarray
        Matrix of statistics.

    """
    n, N = x.shape
    kd = kendall(x)
    numerator = np.sqrt(n) * (kd - threshold)
    Pcc = pcc(x)
    Pc = (kd + 1) / 2
    denominator = 4 * np.sqrt(Pcc - np.power(Pc, 2))
    with np.errstate(divide="ignore"):
        return numerator / denominator


def spearman_statistics(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Calculates a matrix of statistics
    where each statistic has a standard Gaussian distribution N(0,1)
    under the assumption that the Spearman correlation
    between the i and j component of the elliptical random vector
    is equal to the threshold.

    Parameters
    ----------
    x : (n,N) array_like
        Sample of the size n from distribution of the N-dimensional elliptical random vector.

    threshold : float
        The threshold in the interval (-1, 1).

    Returns
    -------
    statistics : (N,N) ndarray
        Matrix of statistics.

    """
    n, N = x.shape
    return np.sqrt(n - 1) * (spearman(x) - threshold)


def partial_statistics(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Calculates a matrix of statistics
    where each statistic has a standard Gaussian distribution N(0,1)
    under the assumption that the partial Pearson correlation
    between the i and j component of the elliptical random vector
    is equal to the threshold.

    Parameters
    ----------
    x : (n,N) array_like
        Sample of the size n from distribution of the N-dimensional elliptical random vector.

    threshold : float
        The threshold in the interval (-1, 1).

    Returns
    -------
    statistics : (N,N) ndarray
        Matrix of statistics.

    """
    n, N = x.shape
    kurt = kurtosis(x)
    return _calc_pearson_stat(partial(x), threshold, kurt, n)
