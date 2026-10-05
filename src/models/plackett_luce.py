"""
Plackett-Luce sampler for converting predicted race positions to a coherent
probability distribution over finishing positions.

Given a predicted finish-position score per driver (lower = faster), the sampler
produces a matrix P where P[i, k] = probability that driver i finishes in position k+1.

By construction:
  - Each row sums to 1  (each driver finishes somewhere)
  - Each column sums to 1 (exactly one driver per position)
  - P(podium) = P[:,0] + P[:,1] + P[:,2] is mathematically valid

Implementation uses the Gumbel-max trick, which is mathematically equivalent to
sequential Plackett-Luce sampling but fully vectorised across simulations:

  score_i = log(strength_i) + Gumbel(0,1)
  ranking = argsort(-scores)

where strength_i = exp(-predicted_position_i / temperature).
"""
from __future__ import annotations

import numpy as np


class PlackettLuceSampler:
    """
    Converts predicted finish positions to a position probability matrix.

    Parameters
    ----------
    temperature : float
        Controls sharpness of the distribution. Lower T = more peaked (confident);
        higher T = more uniform. Default 1.0 is a sensible starting point.
    n_simulations : int
        Number of Monte Carlo draws. 10_000 gives smooth probabilities in ~5ms.
    seed : int
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        n_simulations: int = 10_000,
        seed: int = 42,
    ) -> None:
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if n_simulations < 1:
            raise ValueError(f"n_simulations must be >= 1, got {n_simulations}")
        self.temperature = temperature
        self.n_simulations = n_simulations
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def simulate(self, predicted_positions: np.ndarray) -> np.ndarray:
        """
        Run Plackett-Luce simulation for a single race.

        Parameters
        ----------
        predicted_positions : array-like, shape (n_drivers,)
            Predicted finish position per driver (float). Lower = faster.
            May contain values outside [1, n_drivers] — the relative ordering
            is what matters.

        Returns
        -------
        prob_matrix : np.ndarray, shape (n_drivers, n_drivers)
            prob_matrix[i, k] = P(driver i finishes in position k+1).
        """
        pred = np.asarray(predicted_positions, dtype=float)
        n = len(pred)
        if n == 0:
            return np.empty((0, 0), dtype=float)
        if n == 1:
            return np.ones((1, 1), dtype=float)

        # Strength: exp(-position / T), clipped to avoid underflow
        strengths = np.exp(-np.clip(pred, -50.0, 50.0) / self.temperature)
        strengths = np.maximum(strengths, 1e-10)
        log_strengths = np.log(strengths)  # shape (n,)

        # Gumbel-max trick: draw n_simulations × n_drivers Gumbel samples,
        # add log-strengths, argsort descending → each row is a full ranking.
        gumbels = self._rng.gumbel(0.0, 1.0, size=(self.n_simulations, n))
        scores = log_strengths[np.newaxis, :] + gumbels  # (n_sim, n)
        rankings = np.argsort(-scores, axis=1)  # (n_sim, n); rankings[s,k] = driver who finished k+1

        # Count how many times each driver lands in each position
        counts = np.zeros((n, n), dtype=np.int64)
        for pos in range(n):
            counts[:, pos] = np.bincount(rankings[:, pos], minlength=n)

        return counts / self.n_simulations

    def position_probs(self, predicted_positions: np.ndarray) -> dict[str, np.ndarray]:
        """
        Convenience wrapper — returns the three podium probability arrays.

        Returns
        -------
        dict with keys 'p1', 'p2', 'p3', each shape (n_drivers,).
        """
        mat = self.simulate(predicted_positions)
        return {
            "p1": mat[:, 0],
            "p2": mat[:, 1],
            "p3": mat[:, 2],
        }
