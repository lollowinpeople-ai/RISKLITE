"""Fund look-through: what you really own once ETFs and funds are opened up.

A client holding a world ETF, a US ETF and Apple shares may think they hold three things,
but Apple sits in all three. Look-through multiplies each fund's weight in the portfolio by
each underlying stock's weight in the fund and adds up the exposures.

Holdings data is supplied by the user (issuers publish it as CSV/Excel on their fund
pages). Long format, one row per (fund, holding):

    fund,holding,weight
    SWDA.MI,Apple,0.048
"""
from __future__ import annotations

import itertools

import pandas as pd


def read_holdings_csv(path_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer)
    cols = {c.lower().strip(): c for c in df.columns}
    need = ["fund", "holding", "weight"]
    if any(c not in cols for c in need):
        raise ValueError("Holdings CSV needs the columns 'fund', 'holding' and 'weight'.")
    df = df[[cols[c] for c in need]].copy()
    df.columns = need
    df["fund"] = df["fund"].astype(str).str.strip()
    df["holding"] = df["holding"].astype(str).str.strip()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0).astype(float)
    # accept percentages (sum ~100) or fractions (sum ~1) per fund
    sums = df.groupby("fund")["weight"].transform("sum")
    df.loc[sums > 1.5, "weight"] = df.loc[sums > 1.5, "weight"] / 100
    return df


def look_through(weights: dict[str, float], holdings: pd.DataFrame) -> pd.DataFrame:
    """Underlying exposure per company, aggregated across direct and fund positions.

    Positions not found in `holdings` are treated as direct single-name holdings.
    The unlisted remainder of each fund (its weight beyond the holdings supplied, usually
    the long tail) is reported as "<fund> (other holdings)".
    """
    rows = []
    funds = set(holdings["fund"])
    for pos, w in weights.items():
        if pos in funds:
            h = holdings[holdings["fund"] == pos]
            for _, row in h.iterrows():
                rows.append((row["holding"], pos, w * row["weight"]))
            rest = 1 - h["weight"].sum()
            if rest > 1e-9:
                rows.append((f"{pos} (other holdings)", pos, w * rest))
        else:
            rows.append((pos, "direct", w))
    df = pd.DataFrame(rows, columns=["Holding", "Via", "Exposure"])
    total = df.groupby("Holding")["Exposure"].sum()
    via = df.groupby("Holding")["Via"].apply(lambda s: ", ".join(sorted(set(s))))
    n_sources = df.groupby("Holding")["Via"].nunique()
    out = pd.DataFrame({"Exposure": total, "Held via": via, "Sources": n_sources})
    return out.sort_values("Exposure", ascending=False)


def pairwise_overlap(holdings: pd.DataFrame, funds: list[str] | None = None) -> pd.DataFrame:
    """Overlap between each pair of funds: sum over shared holdings of min(weight_a, weight_b).

    0% means nothing in common; 100% means identical funds.
    """
    funds = funds or sorted(set(holdings["fund"]))
    piv = holdings.pivot_table(index="holding", columns="fund", values="weight",
                               aggfunc="sum").fillna(0.0)
    funds = [f for f in funds if f in piv.columns]
    mat = pd.DataFrame(1.0, index=funds, columns=funds)
    for a, b in itertools.combinations(funds, 2):
        mat.loc[a, b] = mat.loc[b, a] = float(piv[[a, b]].min(axis=1).sum())
    return mat


def hidden_concentration(weights: dict[str, float], holdings: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Names held through more than one route, ranked by total exposure."""
    lt = look_through(weights, holdings)
    lt = lt[~lt.index.str.endswith("(other holdings)")]
    return lt[lt["Sources"] > 1].head(top)
