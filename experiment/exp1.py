import torch
from torch.utils.data.dataloader import Dataset, DataLoader
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ochlv_data import *


exp_config = {
    "unique_name": "exp1_instance1",
    "seq_len_days": 120
}

class MySet(Dataset):

    def __init__(self, df, seq_len, target_cols, start_dt=None, end_dt=None) -> None:
        super().__init__()
        # dt, discrete_index, discrete_mask, continue_value, continue_index, continue_mask, stock_code
        self.datas = df
        self.seq_len = seq_len
        self.target_cols = target_cols
        self.stock_code_mapper = None
        self.start_dt = start_dt
        self.end_dt = end_dt
        self._prepare_fetcher(self.datas, seq_len - 1)
    
    def __getitem__(self, index):
        part = self.datas.iloc[index:index+self.seq_len,:]
        dis_idx = []
        dis_mask = []
        con_value = []
        con_idx = []
        con_mask = []
        for i in range(self.seq_len):
            x = preprcess_func(part.iloc[i,:].to_dict())
            dis_idx.append(x['discrete_index'])
            dis_mask.append(x['discrete_mask'])
            con_value.append(x['continue_value'])
            con_idx.append(x['continue_index'])
            con_mask.append(x['continue_mask'])
        return torch.stack(dis_idx), torch.stack(dis_mask),torch.stack(con_value),torch.stack(con_idx),torch.stack(con_mask), self.datas[self.target_cols].iloc[index+LEN, :].to_numpy()

    def __len__(self):
        return len(self.fetch_index)

    def _prepare_fetcher(self, df: pd.DataFrame, k: int):
        
        # 获取所有唯一的stock_code
        stock_codes = df['stock_code'].unique()
        
        # 按stock_code分组，准备存储所有有效的 (stock_code, dt)
        all_valid = []
        
        # 遍历每个stock_code
        for stock in stock_codes:
            # 获取该股票的所有日期
            stock_df = df[df['stock_code'] == stock]
            # 将dt转为datetime
            dates = pd.to_datetime(stock_df['dt'].astype(str), format='%Y%m%d').values
            dates = np.sort(dates)
            
            # 至少需要k+1天数据
            if len(dates) < (k + 1):
                continue
            
            # 将日期转为set用于快速查找
            date_set = set(dates)
            
            # 找最早的日期
            min_dt = dates[0]
            # 往前推k天
            kdt = min_dt - pd.Timedelta(days=k)
            
            # 过滤 dt >= kdt 的数据
            filtered_dates = [d for d in dates if d >= kdt]
            
            # 检查每个日期是否可以作为T（T到T+k都有数据）
            valid_Ts = []
            for d in filtered_dates:
                # 检查 d, d+1, ..., d+k 是否都存在
                all_exist = True
                for i in range(k + 1):
                    check_date = d + pd.Timedelta(days=i)
                    if check_date not in date_set:
                        all_exist = False
                        break
                if all_exist:
                    valid_Ts.append(d)
            
            # 将有效的T添加到结果中，转回yyyymmdd格式
            for dt_date in valid_Ts:
                dt_int = dt_date.strftime('%Y%m%d')
                if dt_int < self.end_dt and dt_int >= self.start_dt:
                    all_valid.append((stock, dt_int))
        
        # 转换为DataFrame
        self.fetch_index = pd.DataFrame(all_valid, columns=['stock_code', 'dt'])

    def _fetcher(self, index: int):
        row = self.fetch_index.iloc[index]
        return (row['stock_code'], row['dt'])

def preprocess(exp_config):
    pass


def train(exp_config):
    pass


def valid(exp_config):
    pass


