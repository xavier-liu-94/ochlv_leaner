import torch
from torch.utils.data.dataloader import Dataset, DataLoader
import polars as pl
import numpy as np
from datetime import datetime, timedelta
from daily_data.tushare_engine import get_db_file, get_table_name, get_list_df
from .exp1_func import calc_strategy_returns, compute_all_kline_indicators, feature_columns_str, label_column
import duckdb
import os
from tqdm import tqdm


exp_config = {
    "unique_name": "exp1_instance1",
    "seq_len_days": 120,
    "train_start": '20070101',
    "train_end": "20250101",
    "test_start": '20250101',
    "test_end": "20260801"
}

class BinaryTargetsSet(Dataset):

    def __init__(self, seq_len, start_dt=None, end_dt=None, preprocess_func=None) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.hash_id = hash(start_dt) + hash(end_dt) + hash(str(seq_len))
        self.conn = duckdb.connect()
        self._prepare_fetcher()
        self.preprocess_func = preprocess_func
    
    def __getitem__(self, index):
        # return shape seq_feature_tensor: [1, seq_len, feature_nums], mask: [1, seq_len]
        trade_date, ts_code = self.index_df.iloc[index, :].to_list()

        sql = f"""
        WITH date_series AS (
            SELECT 
                STRFTIME(STRPTIME('{trade_date}', '%Y%m%d')::DATE - CAST((idx) AS INTEGER), '%Y%m%d') AS trade_date,
                idx
            FROM generate_series(1, {self.seq_len}) AS t(idx)
        ),
        raw_data AS (
            SELECT *
            FROM read_parquet('tmp_binary_targets_set/*.parquet')
            WHERE ts_code = '{ts_code}'
            AND trade_date >= (SELECT MIN(trade_date) FROM date_series)
            AND trade_date < '{trade_date}'
        ),
        base_close AS (
            SELECT close AS base_close 
            FROM raw_data 
            WHERE close IS NOT NULL 
            ORDER BY trade_date DESC 
            LIMIT 1
        )
        SELECT 
            COALESCE(r.open / b.base_close, 0.0) AS open_rel,
            COALESCE(r.high / b.base_close, 0.0) AS high_rel,
            COALESCE(r.low / b.base_close, 0.0) AS low_rel,
            COALESCE(r.close / b.base_close, 0.0) AS close_rel,
            CASE WHEN r.close IS NOT NULL THEN 1.0 ELSE 0.0 END AS mask,
            *
        FROM date_series d
        LEFT JOIN raw_data r ON d.trade_date = r.trade_date
        CROSS JOIN base_close b
        ORDER BY d.idx;
        """
        
        pl_df = self.conn.execute(sql).pl()

        if self.preprocess_func is None:
            return pl_df, ts_code, trade_date
        
        else:
            return self.preprocess_func(pl_df), ts_code, trade_date

        # feature_cols = feature_columns_str + ['open_rel', 'high_rel', 'low_rel', 'close_rel']

        # data_np = np.column_stack([arrow_table[col].to_numpy() for col in feature_cols])
        # mask_np = arrow_table['mask'].to_numpy()
        # label_np = self.conn.execute(f"select income from read_parquet('tmp_binary_targets_set/*.parquet') where ts_code = '{ts_code}' and trade_date = '{trade_date}'").fetch_arrow_table()['income'].to_numpy()

        # data_tensor = torch.from_numpy(data_np.astype(np.float32))
        # mask_tensor = torch.from_numpy(mask_np.astype(np.float32))
        # label_tensor = torch.from_numpy(label_np.astype(np.float32))



        return data_tensor, mask_tensor, label_tensor, ts_code, trade_date
        
        return data_tensor, mask_tensor, label_tensor, ts_code, trade_date
  
    def __len__(self):
        return len(self.index_df)

    def _prepare_fetcher(self):
        try:
            conn = duckdb.connect()
            self.index_df = conn.execute(f"""
            select trade_date, ts_code
            from 
                (
                    select trade_date, ts_code, min(trade_date) over(partition by ts_code) as min_dt 
                    from read_parquet('tmp_binary_targets_set/*.parquet')
                ) 
            where date_diff('day', strptime(min_dt, '%Y%m%d')::DATE, strptime(trade_date, '%Y%m%d')::DATE) >= {self.seq_len+1}
            and trade_date >= '{self.start_dt}' and trade_date < '{self.end_dt}'
            """).df()
        finally:
            conn.close()

    @staticmethod
    def prepare_full_data():

        tb_name = get_table_name()
        try:
            conn = duckdb.connect(get_db_file())
            all_stock = pl.read_database(f"select distinct ts_code from {tb_name}", connection=conn)
            list_df = get_list_df()
            fold = "tmp_binary_targets_set"
            if os.path.exists(fold):
                os.rmdir(fold)
            os.makedirs(fold, exist_ok=True)
            
            batch_dfs = []
            file_idx = 0
            for i, ts_code in tqdm(enumerate(all_stock['ts_code'].to_list())):
                
                ori_df = pl.read_database(f"select * from {tb_name} where ts_code='{ts_code}'", connection=conn)
                
                start_dt = list_df.filter(pl.col('ts_code') == ts_code)['list_date'][0]
                ori_df = ori_df.filter(pl.col('trade_date') >= start_dt)

                ori_df = ori_df.with_columns(
                    (pl.col('open') * pl.col('adj_factor')).alias('open')
                )
                ori_df = ori_df.with_columns(
                    (pl.col('high') * pl.col('adj_factor')).alias('high')
                )
                ori_df = ori_df.with_columns(
                    (pl.col('low') * pl.col('adj_factor')).alias('low')
                )
                ori_df = ori_df.with_columns(
                    (pl.col('close') * pl.col('adj_factor')).alias('close')
                )

                income_df = calc_strategy_returns(ori_df, dt_col='trade_date', code_col='ts_code')

                indi = compute_all_kline_indicators(income_df)
                income_df = income_df.join(indi, on='trade_date', how='right')
                
                batch_dfs.append(income_df)
                if len(batch_dfs) >= 100:
                    merged_df = pl.concat(batch_dfs)
        
                    file_path = os.path.join(fold, f'dataset_part_{file_idx:04d}.parquet')
                    merged_df.write_parquet(file_path, compression='zstd')
                    
                    del merged_df      
                    batch_dfs = []     
                    file_idx += 1      
            if batch_dfs:
                merged_df = pl.concat(batch_dfs)
                file_path = os.path.join(fold, f'dataset_part_{file_idx:04d}.parquet')
                merged_df.write_parquet(file_path, compression='zstd')
                
                del merged_df
                batch_dfs = []
        finally:
            conn.close()


def preprocess(exp_config):
    from core.field_meta import get_preprocess_function, get_meta_process_info_from_dataframe, save
    import polars as pl

    ds = BinaryTargetsSet(exp_config['seq_len_days'], exp_config['train_start'], exp_config['train_end'])

    def collate_fn_tuple(batch):
        combined_pl_df = pl.concat([x[0] for x in batch], how="vertical")
    
        return combined_pl_df

    dl = DataLoader(ds, 1, shuffle=True, collate_fn=collate_fn_tuple, num_workers=8)
    print(len(ds))
    collected = []
    for i, batch_data in enumerate(dl):
        collected.append(batch_data)
        print(i)
        if i > 200:
            break
    full_data = pl.concat(collected, how="vertical").to_pandas()
    fm, pi = get_meta_process_info_from_dataframe(full_data[feature_columns_str + ['open_rel', 'high_rel', 'low_rel', 'close_rel']])
    os.makedirs(f"run/{exp_config['unique_name']}")
    save(os.path.join(f"run/{exp_config['unique_name']}/fmpi.json"), fm, pi)


def train(exp_config):
    pass


def valid(exp_config):
    pass


