from typing import Callable

import numpy as np
from scipy.stats import norm

from .test_statistics import (
    fechner_statistics,
    kendall_statistics,
    kruskal_statistics,
    partial_statistics,
    pearson_statistics,
    sign_similarity_statistics,
    spearman_statistics,
)


def _calc_p_value(
    x: np.ndarray,
    measure: str,
    threshold: float,
    model: str,
    p_value: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    if measure == "pearson":
        if model == "gaussian":
            return p_value(pearson_statistics(x, "gaussian", threshold))
        else:
            return p_value(pearson_statistics(x, "elliptical", threshold))

    if measure == "sign_similarity":
        return p_value(sign_similarity_statistics(x, threshold))

    if measure == "fechner":
        return p_value(fechner_statistics(x, threshold))

    if measure == "kruskal":
        return p_value(kruskal_statistics(x, threshold))

    if measure == "kendall":
        return p_value(kendall_statistics(x, threshold))

    if measure == "spearman":
        return p_value(spearman_statistics(x, threshold))

    if measure == "partial":
        return p_value(partial_statistics(x, threshold))


def threshold_graph_p_value(
    x: np.ndarray, measure: str, threshold: float, model: str = "elliptical"
) -> np.ndarray:
    """
    Calculates p-values for testing N(N-1)/2 hypotheses of the form:
    H_ij: measure of similarity between the i and j component
    of the random vector <= threshold vs K_ij: measure of similarity
    between the i and j component of the random vector > threshold.

    Parameters
    ----------
    x : (n,N) array_like
        Sample of the size n from distribution of the N-dimensional random vector.

    measure: {"pearson", "sign_similarity", "fechner", "kruskal", "kendall", "spearman", "partial"}
        The measure of similarity relative to which the test is performed.

    threshold : float
        The threshold in the interval (0, 1) for sign similarity
        and in the interval (-1, 1) for other measures.

    model : {"gaussian", "elliptical"}
        The model according to which the random vector is distributed.

    Returns
    -------
    p_value : (N,N) ndarray
        Matrix of p-values.

    """
    p_value = np.vectorize(lambda y: 1 - norm.cdf(y))
    return _calc_p_value(x, measure, threshold, model, p_value)


def concentration_graph_p_value(
    x: np.ndarray, measure: str, model: str = "elliptical"
) -> np.ndarray:
    """
    Calculates p-values for testing N(N-1)/2 hypotheses of the form:
    H_ij: measure of similarity between the i and j component
    of the random vector = 0 vs K_ij: measure of similarity
    between the i and j component of the random vector != 0.

    Parameters
    ----------
    x : (n,N) array_like
        Sample of the size n from distribution of the N-dimensional random vector.

    measure: {"pearson", "sign_similarity", "fechner", "kruskal", "spearman", "partial"}
        The measure of similarity relative to which the test is performed.

    model : {"gaussian", "elliptical"}
        The model according to which the random vector is distributed.

    Returns
    -------
    p_value : (N,N) ndarray
        Matrix of p-values.

    """
    p_value = np.vectorize(lambda y: 2 * (1 - norm.cdf(np.abs(y))))
    if measure == "sign_similarity":
        return _calc_p_value(x, measure, 0.5, model, p_value)
    else:
        return _calc_p_value(x, measure, 0, model, p_value)


def bonferroni(p_value: np.ndarray, a: float) -> np.ndarray:
    """
    Bonferroni procedure for testing M=N(N-1)/2 hypotheses
    with tests of the form:
    1(the hypothesis is rejected) if p_value[i,j]<a_ij,
    0(the hypothesis is accepted) if p_value[i,j]>=a_ij.

    In the Bonferonni procedure a_ij=a/M.
    It is known that for this procedure FWER<=a.

    Parameters
    ----------
    p_value : (N,N) ndarray
        Matrix of p-values.

    a : float
        The boundary of FWER.

    Returns
    -------
    decision_matrix : (N,N) ndarray
        Decision matrix.

    """
    N = p_value.shape[0]
    M = N * (N - 1) / 2
    decision = np.vectorize(lambda y: int(y < a / M))
    decision_matrix = decision(p_value)
    for i in range(N):
        decision_matrix[i][i] = 0
    return decision_matrix


def holm_step_down(p_value: np.ndarray, a: float) -> np.ndarray:
    """
    Holm Step Down procedure for testing M=N(N-1)/2 hypotheses
    with tests of the form:
    1(the hypothesis is rejected) if p_value[i,j]<a_ij,
    0(the hypothesis is accepted) if p_value[i,j]>=a_ij.

    It is known that for this procedure FWER<=a.

    Parameters
    ----------
    p_value : (N,N) ndarray
        Matrix of p-values.

    a : float
        The boundary of FWER.

    Returns
    -------
    decision_matrix : (N,N) ndarray
        Decision matrix.

    """
    N = p_value.shape[0]
    M = N * (N - 1) // 2
    decision_matrix = np.zeros((N, N), dtype=int)
    p_value_array = []
    for i in range(N):
        for j in range(i + 1, N):
            p_value_array.append((p_value[i][j], i, j))
    p_value_array.sort()
    for k in range(M):
        if p_value_array[k][0] >= a / (M - k):
            break
        else:
            decision_matrix[p_value_array[k][1]][p_value_array[k][2]] = 1
            decision_matrix[p_value_array[k][2]][p_value_array[k][1]] = 1
    return decision_matrix


def _hochberg(
    p_value: np.ndarray, a: float, comp: Callable[[float, float, int, int], bool]
) -> np.ndarray:
    N = p_value.shape[0]
    M = N * (N - 1) // 2
    decision_matrix = np.ones((N, N), dtype=int)
    p_value_array = []
    for i in range(N):
        decision_matrix[i][i] = 0
        for j in range(i + 1, N):
            p_value_array.append((p_value[i][j], i, j))
    p_value_array.sort(reverse=True)
    for k in range(M):
        if comp(p_value_array[k][0], a, k, M):
            break
        else:
            decision_matrix[p_value_array[k][1]][p_value_array[k][2]] = 0
            decision_matrix[p_value_array[k][2]][p_value_array[k][1]] = 0
    return decision_matrix


def hochberg_step_up(p_value: np.ndarray, a: float) -> np.ndarray:
    """
    Hochberg Step-up procedure for testing M=N(N-1)/2 hypotheses
    with tests of the form:
    1(the hypothesis is rejected) if p_value[i,j]<a_ij,
    0(the hypothesis is accepted) if p_value[i,j]>=a_ij.

    It is known that for this procedure is proved that FWER<=a
    under assumption of positive dependence of the components
    of the vector X.

    Parameters
    ----------
    p_value : (N,N) ndarray
        Matrix of p-values.

    a : float
        The boundary of FWER.

    Returns
    -------
    decision_matrix : (N,N) ndarray
        Decision matrix.

    """

    def comp(x, a, k, M):
        return x < a / (k + 1)

    return _hochberg(p_value, a, comp)


def benjamini_hochberg(p_value: np.ndarray, a: float) -> np.ndarray:
    """
    Benjamini-Hochberg procedure for testing M=N(N-1)/2 hypotheses
    with tests of the form:
    1(the hypothesis is rejected) if p_value[i,j]<a_ij,
    0(the hypothesis is accepted) if p_value[i,j]>=a_ij.

    Parameters
    ----------
    p_value : (N,N) ndarray
        Matrix of p-values.

    a : float
        The boundary of FWER.

    Returns
    -------
    decision_matrix : (N,N) ndarray
        Decision matrix.

    """

    def comp(x, a, k, M):
        return x <= (M - k) / M * a

    return _hochberg(p_value, a, comp)
