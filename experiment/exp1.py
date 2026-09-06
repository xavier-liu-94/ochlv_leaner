import torch
from torch.utils.data.dataloader import Dataset, DataLoader
import polars as pl
import numpy as np
from datetime import datetime, timedelta
from daily_data.tushare_engine import get_db_file, get_table_name, get_list_df
from .exp1_func import calc_strategy_returns, compute_all_kline_indicators
import duckdb
import os
from tqdm import tqdm


exp_config = {
    "unique_name": "exp1_instance1",
    "seq_len_days": 120
}

class BinaryTargetsSet(Dataset):

    def __init__(self, seq_len, start_dt=None, end_dt=None) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.hash_id = hash(start_dt) + hash(end_dt) + hash(str(seq_len))
        self._prepare_fetcher()
    
    def __getitem__(self, index):
        # return shape seq_feature_tensor: [1, seq_len, feature_nums], mask: [1, seq_len]
        part = self.datas.iloc[index:index+self.seq_len,:]
        dis_idx = []
        dis_mask = []
        con_value = []
        con_idx = []
        con_mask = []
        for i in range(self.seq_len):
            x = preprcess_futqdmnc(part.iloc[i,:].to_dict())
            dis_idx.append(x['discrete_index'])
            dis_mask.append(x['discrete_mask'])
            con_value.append(x['continue_value'])
            con_idx.append(x['continue_index'])
            con_mask.append(x['continue_mask'])
        return seq_feature_tensor, mask

    def __len__(self):
        return len(self.fetch_index)

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
    pass


def train(exp_config):
    pass


def valid(exp_config):
    pass


