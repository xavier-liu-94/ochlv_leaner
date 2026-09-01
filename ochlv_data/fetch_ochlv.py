#!/usr/bin/env python3
"""
OCHLV Data Fetcher

用法:
  python fetch_ochlv.py --mode full                  # 全量拉取,自动断点续传
  python fetch_ochlv.py --mode date --date 20240801  # 拉取指定日期
  python fetch_ochlv.py --mode full --markets A      # 仅拉A股
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
import yaml
import pandas as pd
import tushare as ts

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _today_str():
    return datetime.now().strftime("%Y%m%d")

def _yesterday_str():
    return (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
DATA_DIR = os.path.join(BASE_DIR, "ochlv_data")
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")
META_PATH = os.path.join(DATA_DIR, "stock_meta.parquet")

RATE_LIMIT_SEC = 1.3
DAILY_MAX_RETRIES = 3
META_SAVE_INTERVAL = 10
CHUNK_REST_INTERVAL = 200
CHUNK_REST_SEC = 5

COLUMNS = ["dt", "stock_code", "open", "high", "low", "close", "volume", "turnover_rate"]


def load_config():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["tushare"]["token"]

 
def get_pro():
    token = load_config()
    ts.set_token(token)
    return ts.pro_api()


def get_data_dir(market):
    d = os.path.join(DATA_DIR, market)
    os.makedirs(d, exist_ok=True)
    return d


def year_path(market, year):
    return os.path.join(get_data_dir(market), f"{year}.parquet")


def _normalize_dt_series(s):
    return s.astype(str).str.strip().str.replace("-", "")


def _normalize_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


# ── stock_meta ────────────────────────────────────────────────────────


def load_meta():
    if os.path.exists(META_PATH):
        return pd.read_parquet(META_PATH)
    return pd.DataFrame(columns=["stock_code", "stock_name", "market", "start_dt", "end_dt", "fetched_years"])


def save_meta(df):
    import tempfile
    d = os.path.dirname(META_PATH)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".parquet")
    os.close(fd)
    df.to_parquet(tmp, index=False)
    os.replace(tmp, META_PATH)


def get_completed_years(row):
    s = row.get("fetched_years", "")
    if pd.isna(s) or str(s).strip() == "":
        return set()
    return set(str(x).strip() for x in str(s).split(","))


def mark_year_completed(row, year):
    years = get_completed_years(row)
    years.add(str(year))
    row["fetched_years"] = ",".join(sorted(years))


# ── stock_meta fetching ─────────────────────────────────────────────


def fetch_stock_meta_a(pro):
    fields = "ts_code,symbol,name,list_date,delist_date"
    df_l = pro.stock_basic(exchange="", list_status="L", fields=fields)
    df = df_l
    df = df.drop_duplicates(subset=["ts_code"])
    df = df.rename(columns={"ts_code": "stock_code", "name": "stock_name",
                            "list_date": "start_dt", "delist_date": "end_dt"})
    df["market"] = "A"
    df["start_dt"] = _normalize_dt_series(df["start_dt"])
    df["end_dt"] = _normalize_dt_series(df["end_dt"]).fillna("")
    df["fetched_years"] = ""
    return df[["stock_code", "stock_name", "market", "start_dt", "end_dt", "fetched_years"]]


def fetch_stock_meta_hk(pro):
    df = pro.hk_basic(list_status="L")
    if df is None or df.empty:
        return pd.DataFrame()
    rename_map = {}
    if "ts_code" in df.columns:
        rename_map["ts_code"] = "stock_code"
    if "name" in df.columns:
        rename_map["name"] = "stock_name"
    if "list_date" in df.columns:
        rename_map["list_date"] = "start_dt"
    if "delist_date" in df.columns:
        rename_map["delist_date"] = "end_dt"
    df = df.rename(columns=rename_map)
    for c in ["stock_code", "stock_name", "start_dt", "end_dt"]:
        if c not in df.columns:
            df[c] = ""
    df["market"] = "HK"
    df["start_dt"] = _normalize_dt_series(df["start_dt"])
    df["end_dt"] = _normalize_dt_series(df.get("end_dt", "")).fillna("")
    df["fetched_years"] = ""
    return df[["stock_code", "stock_name", "market", "start_dt", "end_dt", "fetched_years"]]


def fetch_stock_meta(pro, market):
    if market == "A":
        return fetch_stock_meta_a(pro)
    elif market == "HK":
        return fetch_stock_meta_hk(pro)
    else:
        raise ValueError(f"Unknown market: {market}")


# ── OCHLV fetching ───────────────────────────────────────────────────


def fetch_ochlv(pro, stock_code, market, end_dt=None):
    if end_dt is None:
        end_dt = _yesterday_str()
    if market == "A":
        df = pro.daily(ts_code=stock_code, start_date="19900101", end_date=end_dt)
        if df is None or df.empty:
            return None
        df = df[["ts_code", "trade_date", "open", "high", "low", "close", "vol"]]
        df = df.rename(columns={"ts_code": "stock_code", "trade_date": "dt", "vol": "volume"})
    elif market == "HK":
        df = pro.hk_daily(ts_code=stock_code, start_date="19900101", end_date=end_dt)
        if df is None or df.empty:
            return None
        cols = [c for c in ["ts_code", "trade_date", "open", "high", "low", "close", "vol"] if c in df.columns]
        df = df[cols]
        df = df.rename(columns={"ts_code": "stock_code", "trade_date": "dt", "vol": "volume"})
    else:
        return None
    if df is None or df.empty:
        return None
    df["dt"] = _normalize_dt_series(df["dt"])
    _normalize_numeric(df, ["open", "high", "low", "close", "volume"])
    return df


def fetch_turnover(pro, stock_code, market, end_dt=None):
    if end_dt is None:
        end_dt = _yesterday_str()
    if market == "A":
        df = pro.daily_basic(ts_code=stock_code, start_date="19900101", end_date=end_dt,
                             fields="ts_code,trade_date,turnover_rate")
    else:
        return None
    if df is None or df.empty:
        return None
    df = df.rename(columns={"ts_code": "stock_code", "trade_date": "dt"})
    df["dt"] = _normalize_dt_series(df["dt"])
    _normalize_numeric(df, ["turnover_rate"])
    return df[["stock_code", "dt", "turnover_rate"]]


def fetch_daily_for_date(pro, date_str, market):
    if market == "A":
        df_price = pro.daily(trade_date=date_str)
        if df_price is None or df_price.empty:
            return None
        df_price = df_price[["ts_code", "trade_date", "open", "high", "low", "close", "vol"]]
        df_price = df_price.rename(columns={"ts_code": "stock_code", "trade_date": "dt", "vol": "volume"})

        df_basic = pro.daily_basic(trade_date=date_str, fields="ts_code,trade_date,turnover_rate")
        if df_basic is not None and not df_basic.empty:
            df_basic = df_basic.rename(columns={"ts_code": "stock_code", "trade_date": "dt"})
            df = df_price.merge(df_basic[["stock_code", "dt", "turnover_rate"]],
                                on=["stock_code", "dt"], how="left")
        else:
            df = df_price
            df["turnover_rate"] = float("nan")
    elif market == "HK":
        df = pro.hk_daily(trade_date=date_str)
        if df is None or df.empty:
            return None
        cols = [c for c in ["ts_code", "trade_date", "open", "high", "low", "close", "vol"] if c in df.columns]
        df = df[cols]
        df = df.rename(columns={"ts_code": "stock_code", "trade_date": "dt", "vol": "volume"})
        df["turnover_rate"] = float("nan")
    else:
        return None
    if df is None or df.empty:
        return None
    df["dt"] = _normalize_dt_series(df["dt"])
    _normalize_numeric(df, ["open", "high", "low", "close", "volume", "turnover_rate"])
    return df


# ── storage ──────────────────────────────────────────────────────────


def save_year_data(df, market, year):
    if df is None or df.empty:
        return
    path = year_path(market, year)
    save_df = df[COLUMNS].copy()
    save_df["dt"] = save_df["dt"].astype(str)
    save_df["stock_code"] = save_df["stock_code"].astype(str)
    _normalize_numeric(save_df, COLUMNS[2:])

    if os.path.exists(path):
        try:
            existing = pd.read_parquet(path)
        except Exception:
            print(f"\n  (corrupted {path}, rebuilding)")
            existing = pd.DataFrame()
        combined = pd.concat([existing, save_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["dt", "stock_code"], keep="last")
        combined = combined.sort_values(["dt", "stock_code"]).reset_index(drop=True)
    else:
        combined = save_df.sort_values(["dt", "stock_code"]).reset_index(drop=True)

    combined.to_parquet(path, index=False)


# ── mode: full ───────────────────────────────────────────────────────


def _retry(func, *args, max_retries=DAILY_MAX_RETRIES, label=""):
    for attempt in range(max_retries):
        try:
            return func(*args)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                print(f" [FAIL {label}: {e}]", end="")
    return None


def mode_full(pro, markets, refresh_meta=False):
    total_completed_years = 0
    total_new_rows = 0
    end_dt = _yesterday_str()
    end_year = int(end_dt[:4])

    for market in markets:
        print(f"\n{'=' * 60}")
        print(f"  Market: {market}")
        print(f"{'=' * 60}")

        os.makedirs(get_data_dir(market), exist_ok=True)

        old_meta_all = load_meta()
        old_meta = old_meta_all[old_meta_all["market"] == market] if not old_meta_all.empty else pd.DataFrame()

        if not refresh_meta and not old_meta.empty:
            print(f"Using cached stock_meta ({len(old_meta)} stocks)")
            meta = old_meta.copy()
        else:
            print("Fetching stock list from tushare ...")
            new_meta = fetch_stock_meta(pro, market)
            if new_meta.empty:
                print("  No stocks found, skip.")
                continue
            if not old_meta.empty:
                old_idx = old_meta.set_index("stock_code")
                new_idx = new_meta.set_index("stock_code")
                for code in new_idx.index:
                    if code in old_idx.index:
                        new_idx.loc[code, "fetched_years"] = old_idx.loc[code, "fetched_years"]
                meta = new_idx.reset_index()
            else:
                meta = new_meta

        stocks = meta["stock_code"].tolist()
        n_stocks = len(stocks)
        print(f"Total stocks: {n_stocks}\n")

        meta_dirty = False
        batch_size = 100
        batch_data = {}

        def flush_batch():
            if not batch_data:
                return
            for yr, frames in batch_data.items():
                batch_df = pd.concat(frames, ignore_index=True)
                save_year_data(batch_df, market, yr)
            batch_data.clear()

        for i, (idx, row) in enumerate(meta.iterrows()):
            code = row["stock_code"]
            name = row["stock_name"]
            completed = get_completed_years(row)

            start_str = str(row["start_dt"])
            if len(start_str) >= 4:
                start_year = int(start_str[:4])
            else:
                start_year = 1990

            pending_years = set(range(start_year, end_year + 1)) - completed
            total_completed_years += len(set(range(start_year, end_year + 1)) & completed)

            if not pending_years:
                msg = f"[{i + 1:>5}/{n_stocks}] {code} {name}  DONE"
                print(msg, flush=True)
                continue

            msg = f"[{i + 1:>5}/{n_stocks}] {code} {name}  pending: {sorted(pending_years)}"
            print(msg, end="", flush=True)

            df_price = _retry(fetch_ochlv, pro, code, market, end_dt, label="price")
            time.sleep(RATE_LIMIT_SEC)

            if df_price is None or df_price.empty:
                print("  no OCHLV data")
                if (i + 1) % CHUNK_REST_INTERVAL == 0:
                    print(f"  ... chunk rest {CHUNK_REST_SEC}s ...")
                    time.sleep(CHUNK_REST_SEC)
                continue

            df = df_price
            df["turnover_rate"] = float("nan")

            df["year"] = df["dt"].str[:4]
            years_written = []
            data_years = set(int(y) for y in df["year"].dropna().unique())
            for yr in sorted(pending_years):
                if yr in data_years:
                    grp = df[df["year"] == str(yr)]
                    if not grp.empty:
                        batch_data.setdefault(yr, []).append(grp)
                years_written.append(str(yr))

            if years_written:
                for y in years_written:
                    mark_year_completed(row, y)
                meta.loc[idx, "fetched_years"] = row["fetched_years"]
                meta_dirty = True
                total_new_rows += len(df)

            print(f"  {len(df)} rows, years: {years_written}")

            if (i + 1) % batch_size == 0:
                flush_batch()

            if meta_dirty and (i + 1) % META_SAVE_INTERVAL == 0:
                _persist_meta(meta, old_meta_all, market)
                meta_dirty = False

            if (i + 1) % CHUNK_REST_INTERVAL == 0:
                print(f"  ... chunk rest {CHUNK_REST_SEC}s ...")
                time.sleep(CHUNK_REST_SEC)

        flush_batch()
        if meta_dirty:
            _persist_meta(meta, old_meta_all, market)

    print(f"\nDone. Total new rows: {total_new_rows}, previously completed years: {total_completed_years}")


def _persist_meta(meta_df, old_meta_all, current_market):
    save_meta(meta_df)


# ── mode: date ───────────────────────────────────────────────────────


def mode_date(pro, date_str, markets):
    year = int(date_str[:4])
    for market in markets:
        print(f"Market: {market}, Date: {date_str}")
        df = fetch_daily_for_date(pro, date_str, market)
        if df is None or df.empty:
            print(f"  No data")
            continue
        save_year_data(df, market, year)
        print(f"  Saved {len(df)} rows to {market}/{year}.parquet")
        time.sleep(RATE_LIMIT_SEC)


# ── main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="OCHLV Data Fetcher")
    parser.add_argument("--mode", required=True, choices=["full", "date"],
                        help="full: all history with resume; date: specific date")
    parser.add_argument("--date", type=str, default=None,
                        help="Date in YYYYMMDD format (required for date mode)")
    parser.add_argument("--refresh-meta", action="store_true",
                        help="Re-fetch stock list from tushare (instead of using cached)")
    parser.add_argument("--markets", type=str, default="A,HK",
                        help="Comma-separated markets (default: A,HK)")
    args = parser.parse_args()

    if args.mode == "date" and not args.date:
        parser.error("--date is required for date mode")

    markets = [m.strip() for m in args.markets.split(",") if m.strip()]
    pro = get_pro()

    if args.mode == "full":
        mode_full(pro, markets, refresh_meta=args.refresh_meta)
    elif args.mode == "date":
        mode_date(pro, args.date, markets)


if __name__ == "__main__":
    main()
