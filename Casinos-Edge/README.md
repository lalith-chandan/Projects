# Visualising the Central Limit Theorem with a Casino's Roulette

A small simulation project that demonstrates the **Central Limit Theorem (CLT)**
and the **house edge** using a simple roulette betting game.

A player repeatedly bets \$1 on a colour (red or black) on an American roulette
wheel. Each round pays **+\$1 on a win** and **−\$1 on a loss**. Because the wheel
has two green pockets that never win a red/black bet, the probability of winning
is only **18/38 ≈ 0.474**, not 0.5, and that tiny asymmetry is the house edge.

A player's final bankroll after *n* rounds is the sum of *n* independent ±1
outcomes. The CLT predicts that, across many players, those final bankrolls follow
a normal distribution. This project simulates that and checks the result against
the closed-form Gaussian.

## What it shows

1. **A single player's bankroll** over 1000 rounds: a noisy random walk that drifts
   downward over time.
2. **The distribution of 1000 players' final bankrolls**, a histogram that takes on
   a bell shape.
3. **Simulation vs. theory**: the simulated histogram overlaid with the theoretical
   Gaussian, which match closely. The whole distribution sits to the left of zero,
   which is the house edge made visible.

## The math

For a sum of *n* independent ±1 outcomes with win probability *p*:

| Quantity | Formula |
| --- | --- |
| Mean | μ = n · (2p − 1) |
| Variance | σ² = n · 4p(1 − p) |

With *p* = 18/38 and *n* = 1000, the expected final bankroll is about **−\$53**, the
casino's edge averaged over many players.

## Project layout

```
.
├── roulette.py                              # reusable, vectorised simulation functions
├── visualising_clt_using_roulette.ipynb     # the narrative notebook (run top to bottom)
├── requirements.txt
├── LICENSE
└── README.md
```

## Running it

```bash
# (optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate

pip install -r requirements.txt
jupyter notebook visualising_clt_using_roulette.ipynb
```

Then run the cells top to bottom. The simulation is seeded, so you'll get
reproducible plots.

You can also use the simulation functions directly:

```python
import numpy as np
import roulette

rng = np.random.default_rng(42)
p = roulette.win_probability(p_red=0.5, p_black=0.5)
final_bankrolls = roulette.simulate_final_bankrolls(
    n_players=1000, n_games=1000, win_prob=p, rng=rng
)
print(final_bankrolls.mean(), roulette.theoretical_mean(1000, p))
```
