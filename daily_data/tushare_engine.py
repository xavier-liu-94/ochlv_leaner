import tushare as ts
import pandas as pd
from datetime import datetime
import logging
import time
from config import tushare_token
import duckdb
from tqdm import tqdm

logger = logging.getLogger(__name__)

ts.set_token(tushare_token)
pro = ts.pro_api()
conn = duckdb.connect("daily_data/tushare_data.duckdb")


def get_stock_daily_data(stock_code: str, start_date: str = None, end_date: str = None, 
                         step_years: int = 10) -> pd.DataFrame:
    """
    获取指定股票上市至今的日线数据。
    
    参数:
        step_years (int): 每次请求的时间跨度（年）。默认 10 年。
                          (4000条限制约等于16年，设为10年既高效又绝对安全)
    """
    if start_date is None:
        stock_info = pro.stock_basic(ts_code=stock_code, fields='ts_code,list_date')
        if stock_info.empty:
            raise ValueError(f"找不到股票 {stock_code}")
        start_date = stock_info.iloc[0]['list_date']
        logger.debug(f"未指定起始日期，自动获取 {stock_code} 上市日期: {start_date}")

    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')

    start_dt = datetime.strptime(start_date, '%Y%m%d')
    end_dt = datetime.strptime(end_date, '%Y%m%d')

    # 计算总分片数，用于进度条
    total_steps = (end_dt.year - start_dt.year) // step_years + 1
    logger.info(f"开始获取 {stock_code} 数据 (按每次 {step_years} 年分片，预计 {total_steps} 次请求)...")

    def fetch_data_chunked(api_func, data_name, delay: float = 0, **kwargs):
        chunks = []
        current_year = start_dt.year
        end_year = end_dt.year
        current_step = 0
        
        while current_year <= end_year:
            current_step += 1
            
            # 计算当前分片的结束年份
            chunk_end_year = min(current_year + step_years - 1, end_year)
            
            # 精确计算当前分片的起止日期（处理首尾年份不是完整年的情况）
            chunk_start_dt = max(start_dt, datetime(current_year, 1, 1))
            chunk_end_dt = min(end_dt, datetime(chunk_end_year, 12, 31))
            
            current_start_str = chunk_start_dt.strftime('%Y%m%d')
            current_end_str = chunk_end_dt.strftime('%Y%m%d')
            
            progress = (current_step / total_steps) * 100
            logger.debug(f"[{data_name}] 进度: {progress:5.1f}% | 拉取区间: {current_start_str} -> {current_end_str}")

            kwargs['start_date'] = current_start_str
            kwargs['end_date'] = current_end_str
            
            try:
                df_chunk = api_func(**kwargs)
                if not df_chunk.empty:
                    chunks.append(df_chunk)
            except Exception as e:
                logger.warning(f"[{data_name}] 拉取 {current_start_str}-{current_end_str} 时异常: {e}")

            if delay > 0:
                logger.debug(f"⏳ 触发 {data_name} 频率限制保护，休眠 {delay} 秒...")
                time.sleep(delay)

            # 推进到下一个分片
            current_year += step_years

        if chunks:
            return pd.concat(chunks, ignore_index=True)
        return pd.DataFrame()

    # 1. 日线行情 (无严格限制，极短延时)
    logger.debug("-> [1/3] 正在下载日线行情...")
    df_daily = fetch_data_chunked(pro.daily, "日线行情", delay=0.2, ts_code=stock_code)
    if df_daily.empty:
        return pd.DataFrame()

    # 2. 复权因子 (无严格限制，极短延时)
    logger.debug("-> [2/3] 正在下载复权因子...")
    df_factor = fetch_data_chunked(pro.adj_factor, "复权因子", delay=0.2, ts_code=stock_code)

    # 3. 每日基本指标 (受限接口，休眠 61 秒)
    logger.debug("-> [3/3] 正在下载每日基本指标 (受限接口，请耐心等待)...")
    df_basic = fetch_data_chunked(
        pro.daily_basic, "基本指标", delay=0.2, 
        ts_code=stock_code, fields='ts_code,trade_date,turnover_rate'
    )

    # 合并与清洗
    logger.debug("正在合并数据与清洗...")
    df_merged = pd.merge(df_daily, df_factor, on=['ts_code', 'trade_date'], how='left')
    df_merged = pd.merge(df_merged, df_basic, on=['ts_code', 'trade_date'], how='left')
    df_merged = df_merged.sort_values('trade_date').reset_index(drop=True)
    
    target_cols = ['trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount', 'turnover_rate', 'adj_factor']
    existing_cols = [col for col in target_cols if col in df_merged.columns]
    df_result = df_merged[existing_cols].copy()

    logger.debug(f"获取完成！共 {len(df_result)} 条记录。")
    return df_result


def get_all_stocks_ever_listed() -> pd.DataFrame:
    """
    获取 A 股历史上所有上市过的股票列表（包含已退市和暂停上市）。
    
    返回:
        pd.DataFrame: 包含 ts_code, name, list_date, delist_date, list_status 等字段。
    """

    # 定义需要获取的字段
    fields = 'ts_code,symbol,name,area,industry,fullname,enname,cnspell,market,list_date,delist_date,list_status,act_name,act_ent_type'

    logger.debug("-> 正在获取当前上市股票 (L)...")
    df_l = pro.stock_basic(list_status='L', fields=fields)
    
    logger.debug("-> 正在获取已退市股票 (D)...")
    df_d = pro.stock_basic(list_status='D', fields=fields)
    
    logger.debug("-> 正在获取暂停上市股票 (P)...")
    df_p = pro.stock_basic(list_status='P', fields=fields)

    # 合并三个 DataFrame
    df_all = pd.concat([df_l, df_d, df_p], ignore_index=True)

    # 按上市日期升序排序
    df_all = df_all.sort_values(by='list_date').reset_index(drop=True)

    logger.debug(f" 获取完成！共计 {len(df_all)} 只股票 (上市中: {len(df_l)}, 已退市: {len(df_d)}, 暂停: {len(df_p)})")
    return df_all


def _insert_overwrite(start_date="20060101", end_date="20260830"):
    """
    获取所有历史上市股票的日线数据，并按主键 (ts_code, trade_date) 覆盖写入 daily 表。
    幂等：重复运行只会更新/补齐区间内的数据。
    """

    df_stocks = get_all_stocks_ever_listed()
    total = len(df_stocks)
    logger.info(f"开始更新 {total} 只股票日线数据 ({start_date} ~ {end_date})")

    cols = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close',
            'vol', 'amount', 'turnover_rate', 'adj_factor']

    for i, row in tqdm(df_stocks.iterrows()):
        ts_code = row['ts_code']

        # 跳过与目标区间无交集的股票，节省 API 配额
        list_date = row.get('list_date')
        delist_date = row.get('delist_date')
        if pd.notna(list_date) and str(list_date) > end_date:
            continue
        if pd.notna(delist_date) and str(delist_date) < start_date:
            continue

        df = get_stock_daily_data(ts_code, start_date, end_date)
        if df.empty:
            logger.warning(f"[{i + 1}/{total}] {ts_code} 无数据，跳过")
            continue

        df['ts_code'] = ts_code
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        df = df.reindex(columns=cols)

        conn.register('_tmp_daily', df)
        conn.execute("INSERT OR REPLACE INTO daily SELECT * FROM _tmp_daily")
        conn.unregister('_tmp_daily')

        if (i + 1) % 100 == 0 or i + 1 == total:
            logger.info(f"[{i + 1}/{total}] 已更新 {ts_code} ({len(df)} 行)")
            time.sleep(1.0)

    logger.info("全部更新完成")


def _init_db():
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily (
            ts_code       VARCHAR NOT NULL,
            trade_date    DATE    NOT NULL,
            open          DOUBLE,
            high          DOUBLE,
            low           DOUBLE,
            close         DOUBLE,
            vol           BIGINT,
            amount        DOUBLE,
            turnover_rate DOUBLE,
            adj_factor    DOUBLE,
            PRIMARY KEY (ts_code, trade_date)
        )
    """)
    logger.info("init DONE")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING, 
        format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # _init_db()
    # _insert_overwrite()