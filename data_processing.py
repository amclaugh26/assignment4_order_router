"""Functions that build a dataframe with executions and quotes. "Feature Engineering".
By Andrew McLaughlin
Last Updated: 11-28-2025

Functions:
market_hours: enforces only market open orders
load_exec: loads executions and prepares data for joining
load_quotes: loads quotes and prepares for joining



"""
#load libraries
from __future__ import annotations

from datetime import time
from pathlib import Path
from typing import Iterable, List, Optional, Set, Union

import pandas as pd
from tqdm.auto import tqdm

#set variables needed for later
MARKET_START = time(hour=9, minute=30)
MARKET_END = time(hour=16, minute=0)

#limit to market hours
def market_hours(timestamp: pd.Series) -> pd.Series:
    """Return a mask that keeps timestamps within regular market hours."""
    local_time = timestamp.dt.time
    return (local_time >= MARKET_START) & (local_time <= MARKET_END)

#load executions
def load_exec(path: Union[str, Path]) -> pd.DataFrame:
    """Load executions, fix timestamps, and drop after-hour trades"""
    path = Path(path)
    df = pd.read_csv(path)

    #fix timestamps
    timestamp_format = "%Y%m%d-%H:%M:%S.%f"
    df["order_time"] = pd.to_datetime(df["order_time"], format=timestamp_format, errors="coerce")
    df["execution_time"] = pd.to_datetime(df["execution_time"], format=timestamp_format, errors="coerce")

    #dropna and make some columns categorical to same memory
    df = df.dropna(subset=["order_time", "execution_time"])
    df["symbol"] = df["symbol"].astype("category")
    df["side"] = df["side"].astype(str).map({"1": "B", "2": "S"}).fillna(df["side"])
    df = df[df["side"].isin({"B", "S"})]
    df = df[
        market_hours(df["order_time"])
    ]
    #ensure standard datatypes to prevent merge conflicts later on
    df["order_qty"] = df["order_qty"].astype(int)
    df["limit_price"] = df["limit_price"].astype(float)
    df["execution_price"] = df["execution_price"].astype(float)
    return df.reset_index(drop=True)

#load quotes
def load_quotes(
    path: Union[str, Path],
    symbols: Optional[Iterable[str]] = None,
    *,
    use_symbols: bool = True,
    chunk_size: int = 500_000,
) -> pd.DataFrame:
    """load quotes, reading the file piece by piece to prevent memory issues"""
    path = Path(path)
    
    #set columns to retain
    usecols = [
        "ticker",
        "bid_price",
        "ask_price",
        "bid_size",
        "ask_size",
        "sip_timestamp",
    ]
    #define datatypes
    dtype = {
        "ticker": "category",
        "bid_price": float,
        "ask_price": float,
        "bid_size": int,
        "ask_size": int,
    }
    
    #set-up how to read the csv
    reader = pd.read_csv(
        path,
        compression="gzip",
        usecols=usecols,
        dtype=dtype,
        iterator=True,
        chunksize=chunk_size,
        low_memory=False,
    )

    #handle case of using a subset of symbols
    symbol_set: Set[str] | None = None
    if symbols is not None and use_symbols:
        symbol_set = set(symbols)

    #create a loop to handle chunks of data
    filtered_frames: List[pd.DataFrame] = []
    for chunk in tqdm(reader, desc="processing quotes chunks", unit="chunk"):
        chunk["quote_time"] = pd.to_datetime(chunk["sip_timestamp"], unit="ns", errors="coerce")
        chunk = chunk.dropna(subset=["quote_time"])
        chunk = chunk[market_hours(chunk["quote_time"])] #limit to market hours

        if symbol_set is not None:
            chunk = chunk[chunk["ticker"].isin(symbol_set)] #only keep certain symbols if requested

        if chunk.empty:
            continue

        filtered_frames.append(
            chunk[["ticker", "bid_price", "ask_price", "bid_size", "ask_size", "quote_time"]]
        )

    #place loaded quotes into a dataframe to be returned at end
    if filtered_frames:
        df = pd.concat(filtered_frames, ignore_index=True)
    else:
        df = pd.DataFrame(
            columns=["ticker", "bid_price", "ask_price", "bid_size", "ask_size", "quote_time"]
        )

    df = df.sort_values(["ticker", "quote_time"]).reset_index(drop=True)
    return df


#combine quotes and executions data into single dataframe
def exec_with_quotes(
    executions: pd.DataFrame,
    quotes: pd.DataFrame,
) -> pd.DataFrame:
    """Attaches the latest quote before the order was made."""
    #copy data to prevent unintentional editing
    execut = executions.copy()
    qu = quotes.copy()

    #merge dataframes 
    execut["symbol"] = execut["symbol"].astype(str)
    qu["ticker"] = qu["ticker"].astype(str)

    quotes_by_ticker = {
        ticker: group.sort_values("quote_time")
        for ticker, group in qu.groupby("ticker", sort=False)
    }
    merged_chunks: List[pd.DataFrame] = []

    for symbol, group in execut.groupby("symbol", sort=False):
        target_quotes = quotes_by_ticker.get(symbol)
        if target_quotes is None or target_quotes.empty:
            continue

        group_sorted = group.sort_values("order_time")
        merged = pd.merge_asof(
            group_sorted,
            target_quotes,
            left_on="order_time",
            right_on="quote_time",
            direction="backward",
            allow_exact_matches=True,
        )
        merged_chunks.append(merged)

    if not merged_chunks:
        return pd.DataFrame(
            columns=[
                *execut.columns,
                "bid_price",
                "ask_price",
                "bid_size",
                "ask_size",
            ]
        )

    merged = pd.concat(merged_chunks, ignore_index=True)
    merged = merged.drop(columns=["quote_time", "ticker"], errors="ignore")
    merged = merged.dropna(subset=["bid_price", "ask_price", "bid_size", "ask_size"])
    return merged.reset_index(drop=True)

#include a column that measures price improvement
def add_price_improvement(executions: pd.DataFrame) -> pd.DataFrame:
    """Create the price improvement column."""
    is_buy = executions["side"] == "B"
    improvement = executions["limit_price"] - executions["execution_price"]
    executions["price_improvement"] = improvement.where(is_buy, -improvement) #returns the opposite for sell orders
    return executions


#function to execute data preparation using functions above
def prepare_training_data(
    *,
    executions_path: Union[str, Path] = Path("/opt/assignment3/execs_from_fix.csv"),
    quotes_path: Union[str, Path] = Path("/opt/assignment4/quotes_2025-09-10_small.csv.gz"),
    max_symbols: Optional[int] = None,
) -> pd.DataFrame:
    """Creates the merged dataframe for model training"""
    executions = load_exec(executions_path)

    if max_symbols:
        symbols = list(executions["symbol"].value_counts().index[:max_symbols])
    else:
        symbols = list(executions["symbol"].unique())

    quotes = load_quotes(quotes_path, symbols=symbols)
    annotated = exec_with_quotes(executions, quotes)
    annotated = add_price_improvement(annotated)
    return annotated
