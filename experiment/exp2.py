import torch
from torch.utils.data.dataloader import IterableDataset, DataLoader
import polars as pl
import numpy as np
from daily_data.tushare_engine import get_db_file, get_table_name, get_list_df
from .exp1_func import calc_strategy_returns, compute_all_kline_indicators, feature_columns_str, label_column
import os
from tqdm import tqdm
from core.model import TransformerModel, DCN
from transformers import get_cosine_schedule_with_warmup
from core.field_meta import get_preprocess_function, get_embedding_module, get_batch_preprocess_function
from core.utils import TorchTrainingVisualizer
import random
from torch.utils.data import get_worker_info
import multiprocessing as mp
from sklearn.metrics import roc_auc_score
import h5py
import json


exp_config = {
    "unique_name": "exp1_instance2",
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


def _ymd_to_ordinal(ymd_int):
    ymd = np.asarray(ymd_int, dtype=np.int64)
    y = ymd // 10000
    m = (ymd // 100) % 100
    d = ymd % 100
    y = y - (m <= 2)
    era = y // 400
    yoe = y - era * 400
    mp = np.where(m > 2, m - 3, m + 9)
    doy = (153 * mp + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


class BinaryTargetsSet(IterableDataset):

    H5_PATH = "tmp_binary_targets_set.h5"

    def __init__(self, seq_len, start_dt=None, end_dt=None, preprocess_func=None) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.hash_id = hash(start_dt) + hash(end_dt) + hash(str(seq_len))
        self.preprocess_func = preprocess_func
        self._prepare_fetcher()
        self.stop_event = mp.Event()

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        C = len(self.columns)
        seq_len = self.seq_len
        col_index = self.col_index
        close_idx = col_index['close']
        ohlc_idx = np.array(
            [col_index['open'], col_index['high'], col_index['low'], col_index['close']],
            dtype=np.int64,
        )
        out_cols = self.columns + ['open_rel', 'high_rel', 'low_rel', 'close_rel', 'mask']

        code_ids = list(self.group_code_ids)
        random.shuffle(code_ids)
        assigned = [cid for i, cid in enumerate(code_ids) if i % num_workers == worker_id]

        self.stop_event.clear()

        MINIBATCH = 256
        f = h5py.File(self.H5_PATH, 'r')
        try:
            data = f['data']
            dates_ds = f['dates']

            for cid in assigned:
                if self.stop_event.is_set():
                    break
                lo = int(self.offsets[cid])
                hi = lo + int(self.lengths[cid])
                d = dates_ds[lo:hi].astype(np.int64)
                X = data[lo:hi]
                L = X.shape[0]

                close = X[:, close_idx]
                valid_pos = np.where(~np.isnan(close))[0]
                if valid_pos.size:
                    close_ff = close[np.maximum.accumulate(valid_pos)]
                else:
                    close_ff = np.ones(L, dtype=np.float32)
                base_close_all = np.empty(L, dtype=np.float32)
                base_close_all[0] = close_ff[0]
                base_close_all[1:] = close_ff[:-1]

                s0 = int(self.group_start_by_code[cid])
                cnt_c = int(self.group_count_by_code[cid])
                if cnt_c == 0:
                    continue
                s1 = s0 + cnt_c

                rows = self.index_row[s0:s1]
                tgt_dates = self.index_date[s0:s1]
                tgt_inc = self.index_income[s0:s1]
                M = rows.shape[0]

                base_close = base_close_all[rows]
                tgt_ymd = np.char.replace(
                    (np.datetime64('1970-01-01', 'D') + tgt_dates).astype('datetime64[D]').astype('U10'),
                    '-', '',
                )

                for m0 in range(0, M, MINIBATCH):
                    if self.stop_event.is_set():
                        break
                    m1 = min(m0 + MINIBATCH, M)
                    nb = m1 - m0
                    r_sub = rows[m0:m1]
                    td_sub = tgt_dates[m0:m1]
                    bc_sub = base_close[m0:m1]

                    win_start = td_sub - seq_len
                    lo_m = np.searchsorted(d, win_start, side='left')
                    cnt = r_sub - lo_m
                    total = int(cnt.sum())
                    cs = np.zeros(nb + 1, dtype=np.int64)
                    np.cumsum(cnt, out=cs[1:])
                    m_flat = np.repeat(np.arange(nb), cnt)
                    row_flat = np.repeat(lo_m, cnt) + (np.arange(total) - np.repeat(cs[:-1], cnt))
                    cal_pos = d[row_flat] - win_start[m_flat]
                    out_idx = (seq_len - 1) - cal_pos

                    F = np.full((nb, seq_len, C + 5), np.nan, dtype=np.float32)
                    F[:, :, C + 4] = 0.0
                    F[m_flat, out_idx, :C] = X[row_flat]
                    F[m_flat, out_idx, C:C + 4] = X[row_flat][:, ohlc_idx] / bc_sub[m_flat][:, None]
                    F[m_flat, out_idx, C + 4] = 1.0

                    for j in range(nb):
                        if self.stop_event.is_set():
                            break
                        df = pl.from_numpy(F[j], schema=out_cols)
                        if self.preprocess_func is None:
                            yield df, self.codes[cid], tgt_ymd[m0 + j]
                        else:
                            yield self.preprocess_func(df, self.codes[cid], tgt_ymd[m0 + j], float(tgt_inc[m0 + j]))
        finally:
            f.close()
        self.stop_event.set()

    def _prepare_fetcher(self):
        with h5py.File(self.H5_PATH, 'r') as f:
            codes = json.loads(f.attrs['codes'])
            columns = json.loads(f.attrs['columns'])
            offsets = f['offsets'][:]
            lengths = f['lengths'][:]
            dates_all = f['dates'][:]
            income_all = f['income'][:]

        start_ord = int(_ymd_to_ordinal(np.array([int(self.start_dt)], dtype=np.int64))[0])
        end_ord = int(_ymd_to_ordinal(np.array([int(self.end_dt)], dtype=np.int64))[0])

        inrange = (dates_all >= start_ord) & (dates_all < end_ord)
        r = np.nonzero(inrange)[0]
        code_r = np.searchsorted(offsets, r, side='right') - 1
        min_d = dates_all[offsets[code_r]]
        keep = (dates_all[r] - min_d) >= (self.seq_len + 1)
        r = r[keep]
        code_r = code_r[keep]
        inc = income_all[r]
        local = r - offsets[code_r]

        pos_mask = inc >= 0.0314
        neg_mask = ~pos_mask
        n_neg = int(neg_mask.sum())
        if n_neg > 0:
            rng = np.random.default_rng()
            neg_keep = rng.random(n_neg) < 0.01
            keep_mask = pos_mask.copy()
            keep_mask[neg_mask] = neg_keep
        else:
            keep_mask = pos_mask

        code_r = code_r[keep_mask]
        r = r[keep_mask]
        inc = inc[keep_mask]
        local = local[keep_mask]

        order = np.argsort(code_r, kind='stable')
        self.index_date = dates_all[r][order]
        self.index_income = inc[order]
        self.index_row = local[order]

        self.codes = codes
        self.columns = columns
        self.col_index = {c: j for j, c in enumerate(columns)}
        self.offsets = offsets
        self.lengths = lengths

        unique_codes, counts = np.unique(code_r[order], return_counts=True)
        self.group_code_ids = unique_codes.astype(np.int64)
        starts = np.zeros(len(unique_codes), dtype=np.int64)
        if len(unique_codes) > 0:
            starts[1:] = np.cumsum(counts)[:-1]
        self.group_start_by_code = np.full(len(codes), -1, dtype=np.int64)
        self.group_count_by_code = np.zeros(len(codes), dtype=np.int64)
        self.group_start_by_code[unique_codes] = starts
        self.group_count_by_code[unique_codes] = counts

    @staticmethod
    def prepare_full_data():
        import duckdb

        tb_name = get_table_name()
        columns = list(feature_columns_str)
        C = len(columns)

        fold = BinaryTargetsSet.H5_PATH
        if os.path.exists(fold):
            os.remove(fold)

        conn = duckdb.connect(get_db_file())
        try:
            all_stock = pl.read_database(f"select distinct ts_code from {tb_name}", connection=conn)
            list_df = get_list_df()

            f = h5py.File(fold, 'w')
            chunk_rows = 4096
            data = f.create_dataset('data', shape=(0, C), maxshape=(None, C),
                                    dtype='float32', chunks=(chunk_rows, C),
                                    compression='gzip', compression_opts=1, shuffle=True)
            dates = f.create_dataset('dates', shape=(0,), maxshape=(None,),
                                     dtype='int64', chunks=(chunk_rows,),
                                     compression='gzip', compression_opts=1, shuffle=True)
            income_ds = f.create_dataset('income', shape=(0,), maxshape=(None,),
                                         dtype='float64', chunks=(chunk_rows,),
                                         compression='gzip', compression_opts=1, shuffle=True)

            codes = []
            offsets = []
            lengths = []
            n_written = 0
            buf_X, buf_D, buf_I = [], [], []
            buf_rows = 0

            def flush():
                nonlocal n_written, buf_rows
                if not buf_X:
                    return
                X = np.concatenate(buf_X, axis=0)
                D = np.concatenate(buf_D, axis=0)
                I = np.concatenate(buf_I, axis=0)
                add = X.shape[0]
                new_n = n_written + add
                data.resize((new_n, C))
                dates.resize((new_n,))
                income_ds.resize((new_n,))
                data[n_written:new_n] = X
                dates[n_written:new_n] = D
                income_ds[n_written:new_n] = I
                n_written = new_n
                buf_X.clear()
                buf_D.clear()
                buf_I.clear()
                buf_rows = 0

            for ts_code in tqdm(all_stock['ts_code'].to_list()):
                ori_df = pl.read_database(f"select * from {tb_name} where ts_code='{ts_code}'", connection=conn)
                if ori_df.height == 0:
                    continue
                list_row = list_df.filter(pl.col('ts_code') == ts_code)
                if list_row.height == 0:
                    continue
                start_dt = list_row['list_date'][0]
                ori_df = ori_df.filter(pl.col('trade_date') >= start_dt)
                ori_df = ori_df.with_columns([
                    (pl.col('open') * pl.col('adj_factor')).alias('open'),
                    (pl.col('high') * pl.col('adj_factor')).alias('high'),
                    (pl.col('low') * pl.col('adj_factor')).alias('low'),
                    (pl.col('close') * pl.col('adj_factor')).alias('close'),
                ])
                income_df = calc_strategy_returns(ori_df, dt_col='trade_date', code_col='ts_code')
                indi = compute_all_kline_indicators(income_df)
                income_df = income_df.join(indi, on='trade_date', how='right')
                if income_df.height == 0:
                    continue

                X = income_df.select(columns).fill_null(0).to_numpy().astype(np.float32)
                D = _ymd_to_ordinal(income_df['trade_date'].cast(pl.Int64).to_numpy())
                I = income_df['income'].to_numpy().astype(np.float64)

                offsets.append(n_written + buf_rows)
                lengths.append(X.shape[0])
                codes.append(ts_code)
                buf_X.append(X)
                buf_D.append(D)
                buf_I.append(I)
                buf_rows += X.shape[0]
                if buf_rows >= 500000:
                    flush()

            flush()

            f.create_dataset('offsets', data=np.asarray(offsets, dtype=np.int64))
            f.create_dataset('lengths', data=np.asarray(lengths, dtype=np.int64))
            f.attrs['codes'] = json.dumps(codes)
            f.attrs['columns'] = json.dumps(columns)
            f.close()
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
        return dis_idx, dis_mask, con_val, con_idx, con_mask, seq_mask, 1 if income >= 0.0314 else 0

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
        num_warmup_steps=40000,
        num_training_steps=400000
    )
    tv = TorchTrainingVisualizer(os.path.join(f"run/{exp_config['unique_name']}/tf_log"))
    
    for e in range(10):
        print(f"epoch {e}")
        for idx, one_data in tqdm(enumerate(dl)):
            test_info = None
            if idx % 2000 == 0:
                test_out_dict = test(seq_model, dl_test, 100)
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
            if idx % 10000 == 0:
                torch.save(seq_model.state_dict(), os.path.join(f"run/{exp_config['unique_name']}/model-{e}.pth"))
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


