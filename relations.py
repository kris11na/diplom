import numpy as np


def equivalent_correlation_value(
    corr: float, input_type: str, output_type: str
) -> float:
    """
    Calculates by the value of one type correlation
    the equivalent value of another type correlation.

    Remember that relations between Pearson's correlation,
    sign similarity measure, Fechner's correlation, Kruskal's correlation,
    Kendall's correlation works in an elliptical distribution.

    But relations between Spearman's correlation and other correlations
    only works in a Gaussian distribution.

    Parameters
    ----------
    corr : float
        The value of input_type correlation.
    input_type : {"pearson", "sign_similarity", "fechner", "kruskal", "kendall", "spearman"}
        The type of input correlation.
    output_type : {"pearson", "sign_similarity", "fechner", "kruskal", "kendall", "spearman"}
        The type of output correlation.

    Returns
    -------
    corr : float
        The value of output_type correlation.

    """
    if input_type == "pearson":
        if output_type == "pearson":
            return corr
        if output_type == "sign_similarity":
            return 0.5 + (1 / np.pi) * np.arcsin(corr)
        if output_type in ["fechner", "kruskal", "kendall"]:
            return (2 / np.pi) * np.arcsin(corr)
        if output_type == "spearman":
            return (6 / np.pi) * np.arcsin(corr / 2)

    if input_type == "sign_similarity":
        if output_type == "pearson":
            return np.sin(np.pi * (corr - 0.5))
        if output_type == "sign_similarity":
            return corr
        if output_type in ["fechner", "kruskal", "kendall"]:
            return 2 * corr - 1
        if output_type == "spearman":
            return (6 / np.pi) * np.arcsin(np.sin(np.pi * (corr - 0.5)) / 2)

    if input_type in ["fechner", "kruskal", "kendall"]:
        if output_type == "pearson":
            return np.sin(np.pi * corr / 2)
        if output_type == "sign_similarity":
            return (1 + corr) / 2
        if output_type in ["fechner", "kruskal", "kendall"]:
            return corr
        if output_type == "spearman":
            return (6 / np.pi) * np.arcsin(np.sin(np.pi / 2 * corr) / 2)

    if input_type == "spearman":
        if output_type == "pearson":
            return 2 * np.sin((np.pi / 6) * corr)
        if output_type == "sign_similarity":
            return 0.5 + (1 / np.pi) * np.arcsin(2 * np.sin((np.pi / 6) * corr))
        if output_type in ["fechner", "kruskal", "kendall"]:
            return (2 / np.pi) * np.arcsin(2 * np.sin((np.pi / 6) * corr))
        if output_type == "spearman":
            return corr
