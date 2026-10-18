"""Monte Carlo simulation of a monthly-contribution portfolio.

Two return models:
  gbm        lognormal returns with mean and volatility estimated from history
  bootstrap  block bootstrap of real monthly returns (keeps fat tails and short-term momentum)

Usage:
  python mc.py --start 2000 --monthly 100 --years 10 --paths 20000
"""
import argparse
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "sp500_monthly.csv")


def load_returns(path=DATA):
    """Monthly simple returns from a CSV with columns month, close."""
    with open(path) as f:
        closes = np.array([float(r["close"]) for r in csv.DictReader(f)])
    return closes[1:] / closes[:-1] - 1


def gbm_returns(mu, sigma, months, n_paths, rng):
    """Monthly simple returns from a lognormal model; mu and sigma are monthly log-return moments."""
    return np.expm1(rng.normal(mu, sigma, size=(n_paths, months)))


def bootstrap_returns(hist, months, n_paths, rng, block=12):
    """Fixed-length blocks of `block` consecutive historical months, drawn at random and glued together."""
    n_blocks = -(-months // block)
    starts = rng.integers(0, len(hist) - block + 1, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_paths, -1)[:, :months]
    return hist[idx]


def simulate(returns, start, monthly):
    """Wealth paths: start capital, plus `monthly` added at the start of each month. Shape (paths, months+1)."""
    n_paths, months = returns.shape
    wealth = np.empty((n_paths, months + 1))
    wealth[:, 0] = start
    for t in range(months):
        wealth[:, t + 1] = (wealth[:, t] + monthly) * (1 + returns[:, t])
    return wealth


def max_drawdown(wealth):
    peak = np.maximum.accumulate(wealth, axis=1)
    return (wealth / peak - 1).min(axis=1)


def var_cvar(x, level=0.05):
    """Historical VaR and CVaR of outcome x at the given tail level (both returned as outcomes, lower = worse)."""
    q = np.quantile(x, level)
    return q, x[x <= q].mean()


def summary(wealth, invested):
    final = wealth[:, -1]
    ret = final / invested - 1
    var, cvar = var_cvar(ret)
    return {
        "invested": invested,
        "p5": np.percentile(final, 5), "p50": np.percentile(final, 50), "p95": np.percentile(final, 95),
        "p_loss": (final < invested).mean(),
        "var5": var, "cvar5": cvar,
        "median_max_dd": np.median(max_drawdown(wealth)),
    }


def fan_svg(wealth, invested_path, path, title):
    """Percentile fan chart as a standalone SVG (no plotting library needed)."""
    W, H, pad = 720, 360, 50
    bands = np.percentile(wealth, [5, 25, 50, 75, 95], axis=0)
    t = np.arange(wealth.shape[1])
    ymax = bands[-1].max() * 1.05
    X = lambda i: pad + i / t[-1] * (W - 2 * pad)
    Y = lambda v: H - pad - v / ymax * (H - 2 * pad)
    line = lambda ys: " ".join("{:.1f},{:.1f}".format(X(i), Y(v)) for i, v in zip(t, ys))
    area = lambda lo, hi: line(hi) + " " + " ".join("{:.1f},{:.1f}".format(X(i), Y(v)) for i, v in zip(t[::-1], lo[::-1]))
    years = t[-1] // 12
    ticks = "".join('<text x="{:.0f}" y="{}" font-size="11" text-anchor="middle">{}y</text>'.format(X(12 * k), H - pad + 16, k)
                    for k in range(0, years + 1, max(1, years // 10)))
    yt = "".join('<text x="{}" y="{:.0f}" font-size="11" text-anchor="end">{:,.0f}</text>'.format(pad - 6, Y(v) + 4, v)
                 for v in np.linspace(0, ymax, 5))
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">
<rect width="{W}" height="{H}" fill="white"/>
<text x="{pad}" y="24" font-size="14" font-weight="bold">{title}</text>
<polygon points="{a1}" fill="#2563eb" opacity="0.15"/>
<polygon points="{a2}" fill="#2563eb" opacity="0.25"/>
<polyline points="{med}" fill="none" stroke="#1d4ed8" stroke-width="2"/>
<polyline points="{inv}" fill="none" stroke="#6b7280" stroke-width="1.5" stroke-dasharray="5,4"/>
<line x1="{pad}" y1="{b}" x2="{r}" y2="{b}" stroke="#9ca3af"/>
{ticks}{yt}
<text x="{r}" y="44" font-size="11" text-anchor="end">band 5-95% / 25-75%, line = median, dashed = money put in</text>
</svg>""".format(W=W, H=H, pad=pad, title=title, a1=area(bands[0], bands[4]), a2=area(bands[1], bands[3]),
                 med=line(bands[2]), inv=line(invested_path), b=H - pad, r=W - pad, ticks=ticks, yt=yt)
    with open(path, "w") as f:
        f.write(svg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=float, default=2000)
    ap.add_argument("--monthly", type=float, default=100)
    ap.add_argument("--years", type=int, default=10)
    ap.add_argument("--paths", type=int, default=20000)
    ap.add_argument("--block", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    hist = load_returns()
    logr = np.log1p(hist)
    months = 12 * a.years
    invested = a.start + a.monthly * months
    invested_path = a.start + a.monthly * np.arange(months + 1)
    print("History: {} months, mean {:+.2%}/month, vol {:.2%}/month (S&P 500 price index, no dividends)\n".format(
        len(hist), hist.mean(), hist.std(ddof=1)))
    print("{:<10} {:>10} {:>10} {:>10} {:>10} {:>8} {:>9} {:>9} {:>8}".format(
        "model", "invested", "p5", "median", "p95", "P(loss)", "VaR5%", "CVaR5%", "med MDD"))
    for name in ("gbm", "bootstrap"):
        rng = np.random.default_rng(a.seed)
        r = (gbm_returns(logr.mean(), logr.std(ddof=1), months, a.paths, rng) if name == "gbm"
             else bootstrap_returns(hist, months, a.paths, rng, a.block))
        w = simulate(r, a.start, a.monthly)
        s = summary(w, invested)
        print("{:<10} {:>10,.0f} {:>10,.0f} {:>10,.0f} {:>10,.0f} {:>8.1%} {:>+9.1%} {:>+9.1%} {:>+8.1%}".format(
            name, s["invested"], s["p5"], s["p50"], s["p95"], s["p_loss"], s["var5"], s["cvar5"], s["median_max_dd"]))
        fan_svg(w, invested_path, os.path.join(HERE, "docs", "fan_{}.svg".format(name)),
                "{:,.0f} start + {:,.0f}/month for {} years ({} model, {:,} paths)".format(
                    a.start, a.monthly, a.years, name, a.paths))


if __name__ == "__main__":
    main()
