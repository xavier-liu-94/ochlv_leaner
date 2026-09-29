import torch
from torch.utils.data.dataloader import IterableDataset, DataLoader
import polars as pl
import numpy as np
from datetime import datetime, timedelta
from daily_data.tushare_engine import get_db_file, get_table_name, get_list_df
from .exp1_func import calc_strategy_returns, compute_all_kline_indicators, feature_columns_str, label_column
import duckdb
import os
from tqdm import tqdm
from core.model import TransformerModel, DCN
from transformers import get_cosine_schedule_with_warmup
from core.field_meta import get_preprocess_function, get_embedding_module, get_batch_preprocess_function
from core.utils import TorchTrainingVisualizer
import random
from torch.utils.data import get_worker_info
import multiprocessing as mp
import pandas as pd
from sklearn.metrics import roc_auc_score


exp_config = {
    "unique_name": "exp1_instance1",
    "seq_len_days": 120,
    "train_start": '20070101',
    "train_end": "20250101",
    "test_start": '20250101',
    "test_end": "20260801",
    "embedding_size": 2,
    "dcn_out_dim": 128,
    "transformer_layers": 8,
    "attention_heads": 16,
    "transformer_hidden": 512
}


class SeqModel(torch.nn.Module):
    def __init__(self, config_dict, fm, pi) -> None:
        super().__init__()
        self.emb = get_embedding_module(fm, pi, config_dict['embedding_size'], 8)
        self.concat_input_dim = config_dict['embedding_size'] * len(pi.data_column_agg)
        self.dcn_out_dim = config_dict['dcn_out_dim']
        self.dcn = DCN(self.concat_input_dim, output_dim=self.dcn_out_dim)
        self.seq_len = config_dict['seq_len_days']
        self.register_parameter("clf_token", torch.nn.Parameter(torch.randn([1, self.dcn_out_dim])))
        self.tf = TransformerModel(
            self.dcn_out_dim, 
            config_dict['transformer_layers'], 
            config_dict['attention_heads'], 
            config_dict['transformer_hidden'], 
            self.seq_len+1)

        self.fc = torch.nn.Linear(self.dcn_out_dim, 1)
    
    def forward(
        self, 
        discrete_index,
        discrete_mask,  
        continue_value, 
        continue_index,
        continue_mask,
        seq_mask
    ):
        bs = discrete_index.size()[0]
        hidden, mask = self.emb(
            discrete_index.view([bs*self.seq_len, -1]), 
            discrete_mask.view([bs*self.seq_len, -1]), 
            continue_value.view([bs*self.seq_len, -1]), 
            continue_index.view([bs*self.seq_len, -1]), 
            continue_mask.view([bs*self.seq_len, -1])
        )
        # [batch*seq_len, feature_num, embedding_size] -> [batch*seq_len, feature_num*embedding_size]
        input_hidden = (hidden * mask.unsqueeze(-1)).view([bs*self.seq_len, self.concat_input_dim])
        hidden = self.dcn(input_hidden)

        # [batch*seq_len, dcn_out_dim] -> [batch, seq_len, dcn_out_dim]
        seq_hidden = torch.cat([hidden.view([bs, self.seq_len, self.dcn_out_dim]), self.clf_token.repeat(bs, 1, 1)], dim=1)
        seq_mask = torch.cat([seq_mask, torch.ones([bs, 1], device=seq_mask.device)], dim=1)

        final_out = self.tf(seq_hidden, seq_mask)

        return self.fc(final_out[:,-1,:])


class BinaryTargetsSet(IterableDataset):

    def __init__(self, seq_len, start_dt=None, end_dt=None, preprocess_func=None) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.hash_id = hash(start_dt) + hash(end_dt) + hash(str(seq_len))
        self._conn = None
        self.preprocess_func = preprocess_func
        self._prepare_fetcher()
        self.stop_event = mp.Event()
    
    @property
    def conn(self):
        if self._conn is None:
            self._conn = duckdb.connect()
        return self._conn

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        ts_code_order = self.index_df['ts_code'].unique()
        random.shuffle(ts_code_order)

        assigned = [ts for i, ts in enumerate(ts_code_order) if i % num_workers == worker_id]
        self.stop_event.clear()

        for ts_code in assigned:
            if self.stop_event.is_set():
                break
            pl_full_df = self.conn.execute(f"""
            SELECT *
            FROM read_parquet('tmp_binary_targets_set/*.parquet')
            WHERE ts_code = '{ts_code}'
            """).pl()
            trade_date_list = self.index_df[self.index_df['ts_code']==ts_code]['trade_date'].unique()
            for trade_date in trade_date_list:
                if self.stop_event.is_set():
                    break
                start_dt = (datetime.strptime(trade_date, '%Y%m%d') - timedelta(days=self.seq_len)).strftime('%Y%m%d')
        
                sub_df = pl_full_df.filter(
                    (pl.col('trade_date') >= start_dt) & (pl.col('trade_date') < trade_date)
                )
                
                valid_close = sub_df.filter(pl.col('close').is_not_null()).sort('trade_date', descending=True)
                base_close = valid_close['close'][0] if len(valid_close) > 0 else 1.0 
                
                pl_df = sub_df.with_columns([
                    (pl.col('open') / base_close).alias('open_rel'),
                    (pl.col('high') / base_close).alias('high_rel'),
                    (pl.col('low') / base_close).alias('low_rel'),
                    (pl.col('close') / base_close).alias('close_rel'),
                    pl.when(pl.col('close').is_not_null()).then(1.0).otherwise(0.0).alias('mask')
                ]).fill_null(0.0)

                date_series = pl.DataFrame({
                    'trade_date': [(datetime.strptime(trade_date, '%Y%m%d') - timedelta(days=i)).strftime('%Y%m%d') for i in range(1, self.seq_len + 1)],
                    'idx': list(range(1, self.seq_len + 1))
                })

                joined = date_series.join(pl_df, on='trade_date', how='left')

                if self.preprocess_func is None:
                    yield joined, ts_code, trade_date
                
                else:
                    high = pl_full_df.filter(pl.col('trade_date') == trade_date)['high'][0]
                    open = pl_full_df.filter(pl.col('trade_date') == trade_date)['open'][0]
                    yield self.preprocess_func(joined, ts_code, trade_date, high/open - 1)
        self.stop_event.set()

    def _prepare_fetcher(self):
        try:
            conn = duckdb.connect()
            self.index_df = conn.execute(f"""
                select trade_date, ts_code,income
                from 
                    (
                        select trade_date, ts_code,income, min(trade_date) over(partition by ts_code) as min_dt 
                        from read_parquet('tmp_binary_targets_set/*.parquet')
                    ) 
                where date_diff('day', strptime(min_dt, '%Y%m%d')::DATE, strptime(trade_date, '%Y%m%d')::DATE) >= {self.seq_len+1}
                and trade_date >= '{self.start_dt}' and trade_date < '{self.end_dt}'
            """).df()
        finally:
            conn.close()
        # mask = self.index_df["income"] >= 0
        # self.index_df = pd.concat([
        #     self.index_df[mask],                              
        #     self.index_df[~mask].sample(frac=0.1)             
        # ], ignore_index=True)

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

    dl = DataLoader(ds, 1, collate_fn=collate_fn_tuple, num_workers=8)

    collected = []
    for i, batch_data in tqdm(enumerate(dl)):
        collected.append(batch_data)
        if i > 2000:
            break
    full_data = pl.concat(collected, how="vertical").to_pandas()
    fm, pi = get_meta_process_info_from_dataframe(full_data[feature_columns_str + ['open_rel', 'high_rel', 'low_rel', 'close_rel']])
    os.makedirs(f"run/{exp_config['unique_name']}", exist_ok=True)
    save(os.path.join(f"run/{exp_config['unique_name']}/fmpi.json"), fm, pi)


def train(exp_config):
    from core.field_meta import load, get_batch_preprocess_function
    fm, pi = load(f"run/{exp_config['unique_name']}/fmpi.json")

    process_batch = get_batch_preprocess_function(fm, pi)

    def func(pl_df: pl.DataFrame, ts_code, trade_date, income):
        dis_idx, dis_mask, con_val, con_idx, con_mask = process_batch(pl_df)
        seq_mask = torch.LongTensor(pl_df["mask"].fill_null(0.0).to_numpy().copy())
        return dis_idx, dis_mask, con_val, con_idx, con_mask, seq_mask, 1 if income >= 0.02 else 0

    ds = BinaryTargetsSet(exp_config['seq_len_days'], exp_config['train_start'], exp_config['train_end'], func)
    dl = DataLoader(ds, batch_size=100, num_workers=4)
    ds_test = BinaryTargetsSet(exp_config['seq_len_days'], exp_config['test_start'], exp_config['test_end'], func)
    dl_test = DataLoader(ds_test, batch_size=100, num_workers=2)

    seq_model = SeqModel(exp_config, fm, pi)
    seq_model.cuda()
    loss_m = torch.nn.BCEWithLogitsLoss()
    loss_m.cuda()
    opti = torch.optim.AdamW(seq_model.parameters(), lr=1e-4)
    sched = get_cosine_schedule_with_warmup(
        opti,
        num_warmup_steps=10000,
        num_training_steps=100000
    )
    tv = TorchTrainingVisualizer(os.path.join(f"run/{exp_config['unique_name']}/tf_log"))
    
    for e in range(10):
        print(f"epoch {e}")
        for idx, one_data in tqdm(enumerate(dl)):
            test_info = None
            if idx % 500 == 0:
                test_out_dict = test(seq_model, dl_test, 20)
                test_info = test_out_dict
            opti.zero_grad()
            out = seq_model(
                one_data[0].cuda(), 
                one_data[1].cuda(), 
                one_data[2].cuda(), 
                one_data[3].cuda(), 
                one_data[4].cuda(), 
                one_data[5].cuda())
            loss = loss_m(out.squeeze(1), one_data[6].float().cuda())
            if test_info is None:
                tv.log_metrics({"loss": loss.detach().cpu().numpy(),"lr": sched.get_last_lr()[0]})
            else:
                tv.log_metrics({"loss": loss.detach().cpu().numpy(),"lr": sched.get_last_lr()[0], 'auc': test_info['auc']})
            loss.backward()
            torch.nn.utils.clip_grad_norm_(seq_model.parameters(), max_norm=1.0)
            opti.step()
            sched.step()
        torch.save(seq_model.state_dict(), os.path.join(f"run/{exp_config['unique_name']}/model-{e}.pth"))


def test_one(model, one_data):
    model.eval()
    full_pred = []
    full_gt = []
    with torch.no_grad():
        out = model(one_data[0].cuda(), one_data[1].cuda(), one_data[2].cuda(), one_data[3].cuda(), one_data[4].cuda(), one_data[5].cuda())
        full_pred.append(torch.sigmoid(out.squeeze(1)).cpu().numpy())
        full_gt.append(one_data[6].numpy())
    model.train()
    auc = roc_auc_score(np.concatenate(full_gt), np.concatenate(full_pred))
    return {'auc': auc}


def test(model, dataloader, max_steps=None):
    model.eval()
    full_pred = []
    full_gt = []
    with torch.no_grad():
        for idx, one_data in tqdm(enumerate(dataloader)):
            out = model(one_data[0].cuda(), one_data[1].cuda(), one_data[2].cuda(), one_data[3].cuda(), one_data[4].cuda(), one_data[5].cuda())
            full_pred.append(torch.sigmoid(out.squeeze(1)).cpu().numpy())
            full_gt.append(one_data[6].numpy())
            if max_steps is not None and idx >= max_steps:
                break
    model.train()
    auc = roc_auc_score(np.concatenate(full_gt), np.concatenate(full_pred))
    return {'auc': auc}

def valid(exp_config):
    pass


