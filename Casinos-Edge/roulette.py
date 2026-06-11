"""Roulette / Central Limit Theorem simulation utilities.

We model a simple, *fair-looking* roulette game on an American wheel
(38 pockets: 18 red, 18 black, 2 green). The player bets $1 on a colour:

    * win  -> net +$1
    * lose -> net -$1

Because the two green pockets never pay out on a red/black bet, the
probability of winning is only 18/38 < 0.5. That tiny asymmetry is the
"house edge", and it is what makes the casino win in the long run even
though each individual round looks like a coin flip.

A player's final bankroll after ``n`` rounds is the sum of ``n``
independent +/-1 outcomes. By the Central Limit Theorem, the
distribution of that sum across many players converges to a normal
distribution whose mean and variance we can write down in closed form.
The functions below let you simulate that and compare it against the
theoretical Gaussian.
"""

from __future__ import annotations

import numpy as np

# American roulette wheel layout.
RED_POCKETS = 18
BLACK_POCKETS = 18
GREEN_POCKETS = 2
TOTAL_POCKETS = RED_POCKETS + BLACK_POCKETS + GREEN_POCKETS  # 38


def win_probability(p_red: float = 0.5, p_black: float = 0.5, p_green: float = 0.0) -> float:
    """Probability of winning a single round.

    ``p_red``, ``p_black`` and ``p_green`` are how often the *player*
    chooses to bet on each colour and must sum to 1. A bet wins only if
    the ball lands on the colour that was chosen, so we weight each
    colour's wheel odds by how often the player picks it.
    """
    total = p_red + p_black + p_green
    if not np.isclose(total, 1.0):
        raise ValueError(f"Bet probabilities must sum to 1, got {total}.")
    return (
        p_red * (RED_POCKETS / TOTAL_POCKETS)
        + p_black * (BLACK_POCKETS / TOTAL_POCKETS)
        + p_green * (GREEN_POCKETS / TOTAL_POCKETS)
    )


def _outcomes(shape, win_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Return an array of +1 (win) / -1 (loss) outcomes of the given shape."""
    wins = rng.random(shape) < win_prob
    return np.where(wins, 1, -1)


def simulate_bankroll(n_games: int, win_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Simulate one player's bankroll trajectory over ``n_games`` rounds.

    Returns an array of length ``n_games + 1`` that starts at 0 (before
    any bet) and records the running bankroll after each round.
    """
    steps = _outcomes(n_games, win_prob, rng)
    return np.concatenate(([0], np.cumsum(steps)))


def simulate_final_bankrolls(
    n_players: int, n_games: int, win_prob: float, rng: np.random.Generator
) -> np.ndarray:
    """Simulate many players and return each one's *final* bankroll.

    This is the quantity the Central Limit Theorem speaks about: each
    final bankroll is a sum of ``n_games`` independent +/-1 outcomes, so
    across ``n_players`` the values are approximately normally
    distributed.
    """
    steps = _outcomes((n_players, n_games), win_prob, rng)
    return steps.sum(axis=1)


def theoretical_mean(n_games: int, win_prob: float) -> float:
    """Expected final bankroll: ``n_games * (2p - 1)``.

    A single round has expectation ``(+1) * p + (-1) * (1 - p) = 2p - 1``.
    """
    return n_games * (2 * win_prob - 1)


def theoretical_variance(n_games: int, win_prob: float) -> float:
    """Variance of the final bankroll: ``n_games * 4p(1 - p)``.

    For a +/-1 variable, ``Var = E[X^2] - E[X]^2 = 1 - (2p - 1)^2 = 4p(1 - p)``.
    """
    return n_games * 4 * win_prob * (1 - win_prob)


def theoretical_std(n_games: int, win_prob: float) -> float:
    """Standard deviation of the final bankroll."""
    return np.sqrt(theoretical_variance(n_games, win_prob))
