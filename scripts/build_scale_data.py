"""Build the SKU/store scale layer from raw M5 data.

Downloads the public M5 dataset (via the datasetsforecast mirror — no Kaggle
auth) and precomputes, over the same window as the portfolio backtest
(2013-05-05 → 2014-01-22):

  data/scale/storeSummary.json   per store: volume, seasonal-naive accuracy
  data/scale/storeDept.json      per store × department: naive accuracy
  data/scale/skuIndex.json       per SKU at CA_1: volume, zero-day share,
                                 naive accuracy (3,049 rows)

Scope is honest by construction: the portfolio's *model* forecasts exist only
for CA_1's seven departments, so the scale layer serves actuals and the
seasonal-naive baseline — metrics computable from raw data alone.

Usage: python -m scripts.build_scale_data   (downloads ~450MB on first run)
"""

import json
from pathlib import Path

import pandas as pd
from datasetsforecast.m5 import M5

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "scale"
WINDOW = ("2013-05-05", "2014-01-22")  # the portfolio backtest window


def naive_wape(g: pd.DataFrame) -> float | None:
    """Seasonal-naive (lag-7) WAPE over the window, in percent."""
    actual = g["y"].to_numpy()
    naive = g["naive"].to_numpy()
    denom = actual.sum()
    if denom == 0:
        return None
    return round(float(100 * abs(actual - naive).sum() / denom), 1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading M5 (downloads on first run)...")
    Y_df, _, S_df = M5.load(directory=str(ROOT / ".m5cache"))
    Y_df["ds"] = pd.to_datetime(Y_df["ds"])

    meta = S_df[["unique_id", "item_id", "dept_id", "store_id", "state_id"]]
    df = Y_df.merge(meta, on="unique_id")
    print(f"{len(df):,} rows loaded")

    def agg_series_metrics(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        """Aggregate to daily series per key FIRST, then lag-7 naive + WAPE.

        Computing naive error on pooled SKU-day rows overstates error wildly
        (SKU-level noise doesn't cancel); store/dept accuracy is only
        meaningful on the aggregated daily series — the same granularity the
        portfolio's planning metrics use.
        """
        daily = frame.groupby(keys + ["ds"], observed=True)["y"].sum().reset_index()
        daily = daily.sort_values(keys + ["ds"])
        daily["naive"] = daily.groupby(keys, observed=True)["y"].shift(7)
        daily = daily[(daily["ds"] >= WINDOW[0]) & (daily["ds"] <= WINDOW[1])].dropna(subset=["naive"])
        return daily

    # ---- per store (aggregated store-day series)
    store_daily = agg_series_metrics(df, ["store_id", "state_id"])
    stores = []
    for (store, state), g in store_daily.groupby(["store_id", "state_id"], observed=True):
        if len(g) == 0:
            continue
        stores.append(
            {
                "store": str(store),
                "state": str(state),
                "volume_units": int(g["y"].sum()),
                "naive_fa_pct": round(100 - naive_wape(g), 1),
            }
        )
    stores.sort(key=lambda r: -r["volume_units"])
    (OUT / "storeSummary.json").write_text(json.dumps(stores, indent=1))

    # ---- per store × dept (aggregated store-dept-day series)
    sd_daily = agg_series_metrics(df, ["store_id", "dept_id"])
    sd = []
    for (store, dept), g in sd_daily.groupby(["store_id", "dept_id"], observed=True):
        if len(g) == 0:
            continue
        sd.append(
            {
                "store": str(store),
                "dept": str(dept),
                "volume_units": int(g["y"].sum()),
                "naive_fa_pct": round(100 - naive_wape(g), 1),
            }
        )
    (OUT / "storeDept.json").write_text(json.dumps(sd, indent=1))

    # ---- per SKU at CA_1 (SKU-day granularity — the SKU's own series)
    df = df.sort_values(["unique_id", "ds"])
    df["naive"] = df.groupby("unique_id", observed=True)["y"].shift(7)
    df = df[(df["ds"] >= WINDOW[0]) & (df["ds"] <= WINDOW[1])].dropna(subset=["naive"])
    ca1 = df[df["store_id"] == "CA_1"]
    skus = []
    for item, g in ca1.groupby("item_id", observed=True):
        wape = naive_wape(g)
        skus.append(
            {
                "item": str(item),
                "dept": str(g["dept_id"].iloc[0]),
                "volume_units": int(g["y"].sum()),
                "zero_day_pct": round(100 * (g["y"] == 0).mean(), 1),
                "naive_fa_pct": None if wape is None else round(100 - wape, 1),
            }
        )
    skus.sort(key=lambda r: -r["volume_units"])
    (OUT / "skuIndex.json").write_text(json.dumps(skus, indent=1))

    print(f"wrote {len(stores)} stores, {len(sd)} store-depts, {len(skus)} SKUs to {OUT}")


if __name__ == "__main__":
    main()
