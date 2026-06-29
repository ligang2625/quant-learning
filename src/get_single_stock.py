import numpy as np
from pathlib import Path
import pandas as pd
import akshare as ak

project_root = project_root = Path(__file__).resolve().parent.parent
RAW_DIR = project_root / "data" / "raw"
PROCESSED_DIR = project_root / "data" / "processed"
REPORT_DIR = project_root / "reports"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def to_sina_symbol(symbol: str) -> str:
    """
    将 6 位 A 股代码转换为新浪格式。
    000001 -> sz000001
    600519 -> sh600519
    """
    symbol = str(symbol).zfill(6)

    if symbol.startswith(("6", "5", "9")):
        return "sh" + symbol
    else:
        return "sz" + symbol


def get_single_stock(
    symbol: str,
    period: str = "daily",
    start_date: str = "20210101",
    end_date: str | None = None,
    adjust: str = "qfq",
    save_to_loacl: bool = True
) -> pd.DataFrame:
    """
    获取 A 股日线数据。
    优先使用东方财富 stock_zh_a_hist；
    如果失败，自动切换到新浪 stock_zh_a_daily。
    """
    symbol = str(symbol).zfill(6)

    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y%m%d")

    try:
        print(f"尝试使用东方财富接口获取 {symbol}...")

        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period=period,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
            timeout=15,
        )

        if df.empty:
            raise ValueError("东方财富接口返回空数据")

        print(f"东方财富接口成功：{symbol}")
        if save_to_loacl:
            df.to_csv(output_path = RAW_DIR / f"{symbol}_{adjust}_{period}.csv", index=False, encoding="utf-8-sig")
        return df

    except Exception as e:
        print(f"东方财富接口失败：{symbol}")
        print(type(e).__name__, e)
        print(f"尝试切换到新浪接口获取 {symbol}...")

        sina_symbol = to_sina_symbol(symbol)

        df = ak.stock_zh_a_daily(
            symbol=sina_symbol,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )

        if df.empty:
            raise ValueError(f"新浪接口也返回空数据：{symbol}")

        # 新浪接口没有股票代码列，补一列
        df["symbol"] = symbol
        if save_to_loacl:
            df.to_csv(output_path = RAW_DIR / f"{symbol}_{adjust}_{period}.csv", index=False, encoding="utf-8-sig")
        print(f"新浪接口成功：{symbol}")
        return df


# =========================
# 数据清洗函数
# =========================
def clean_stock_data(
    df: pd.DataFrame,
    symbol: str,
    save_to_local: bool = True,
) -> pd.DataFrame:
    data = df.copy()

    rename_map = {
        "日期": "date",
        "股票代码": "symbol",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "换手率": "turnover",
    }

    data = data.rename(columns=rename_map)

    # 如果数据里没有 symbol 列，则手动添加
    if "symbol" not in data.columns:
        data["symbol"] = symbol

    # 股票代码统一成字符串，避免 000001 变成 1
    data["symbol"] = data["symbol"].astype(str).str.zfill(6)

    # 日期转换
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    required_cols = ["date", "symbol", "open", "high", "low", "close"]
    missing = [col for col in required_cols if col not in data.columns]

    if missing:
        raise ValueError(f"缺少必要字段：{missing}")

    keep_cols = [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "turnover",
    ]

    # 有些数据源可能没有 turnover，所以这里只保留实际存在的列
    keep_cols = [col for col in keep_cols if col in data.columns]
    data = data[keep_cols]
    
    if save_to_local:
        data.to_csv(output_path = PROCESSED_DIR / f"{symbol}_clean.csv", index=False, encoding="utf-8-sig")
    return data



if __name__ == "__main__":
    RAW_DIR = Path("../data/raw")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    symbol = "000001"

    df = get_single_stock(
        symbol=symbol,
        start_date="20210101",
        adjust="qfq",
    )

    print(df.head())
    print(df.tail())
    print(df.columns)
    print(df.info())

    output_path = RAW_DIR / f"{symbol}_qfq_daily.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"数据已保存到: {output_path}")