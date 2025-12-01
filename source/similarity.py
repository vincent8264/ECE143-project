import numpy as np


def CosineSimilarity(s1, s2):
    """
    Compute the cosine similarity between a matrix of vectors, and a target vector.
    Args:
    - s1 (np.ndarray): 2D matrix of vectors. shape: (n, d), where n is the number of songs and d is the number of features.
    - s2 (np.ndarray): The 1D target vector. shape: (d,)

    Returns:
    - np.ndarray: A 1D array of similarity scores. shape: (n,)
    """

    numer = np.dot(s1, s2)
    denom = np.linalg.norm(s1, axis=1) * np.linalg.norm(s2)

    # Division with zero handling
    similarity = np.divide(
        numer,
        denom,
        out=np.zeros_like(numer, dtype=float),
        where=denom != 0
    )

    return similarity


def JaccardSimilarity(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom > 0:
        return numer / denom
    return 0
