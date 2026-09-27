import numpy as np
import pandas as pd
import torch
import json


class DiscreteCollector():

    def __init__(self, verbose=False) -> None:
        self.field_dict = {}
        self.embedding_num = 0
        self.verbose = verbose

    def record_field_value(self, field_name, value) -> None:
        field_name = str(field_name)
        value = str(value)
        if field_name not in self.field_dict:
            self.field_dict[field_name] = {}
        if value not in self.field_dict[field_name]:
            self.field_dict[field_name][value] = self.embedding_num
            self.embedding_num += 1
    
    def get_embedding_idx(self, field_name, value) -> int:
        field_name = str(field_name)
        value = str(value)
        if field_name not in self.field_dict:
            self._log(f"field_name({field_name}) not in field_dict")
            return self.embedding_num
        if value is not None and value in self.field_dict[field_name]:
            return self.field_dict[field_name][value]
        else:
            return self.embedding_num
    
    def __len__(self) -> int:
        return self.embedding_num

    def _log(self, str):
        if self.verbose:
            print("[WARN] DiscreteCollector: "+str)

class ProcessInfo:

    def __init__(self) -> None:
        # dict {str: str}
        self.data_column_field_name_mapping = None

        # dict {str: value} 每列只允许使用一个值作为空标记
        self.data_column_default_value = None

        # [[str], str] 表明哪些列取完embedding需要加在一起
        self.data_column_agg = None


def _calculate_overall_mean_std(n_list, mean_list, std_list):
    """
    计算整体均值和整体标准差（合并标准差）
    
    参数:
        n_list (list): 每个样本的样本数（如 [10, 20, 30]）
        mean_list (list): 每个样本的均值（如 [5.0, 6.0, 7.0]）
        std_list (list): 每个样本的标准差（如 [1.0, 1.5, 2.0]）
    
    返回:
        tuple: (overall_mean, overall_std)
    """
    # 转换为 NumPy 数组以便向量化计算
    n = np.array(n_list)
    means = np.array(mean_list)
    stds = np.array(std_list)
    
    # 1. 计算整体均值（加权平均）
    overall_mean = np.sum(n * means) / np.sum(n)
    
    # 2. 计算整体方差（合并方差公式）
    # 方差 = (sum((n_i-1)*std_i^2) + sum(n_i*(mean_i - overall_mean)^2)) / (sum(n_i) - 1)
    sum_part1 = np.sum((n - 1) * stds**2)  # 组内方差贡献
    sum_part2 = np.sum(n * (means - overall_mean)**2)  # 组间方差贡献
    overall_var = (sum_part1 + sum_part2) / (np.sum(n) - 1)
    
    # 3. 计算整体标准差
    overall_std = np.sqrt(overall_var)
    
    return (overall_mean, overall_std)


class FieldMeta:

    def __init__(
        self, 
        discrete_fields: list, 
        continuous_fields: list,
        process_info: ProcessInfo,
        dataframe,
        init=True
    ) -> None:
        self.discrete_fields = discrete_fields
        self.continuous_fields = continuous_fields
        self.discrete_collector = DiscreteCollector()
        self.continuous_mean_std = {}
        self.continuous_field_order = []
        if init:
            self._init_values(process_info, dataframe)
    
    def _init_values(self, process_info: ProcessInfo, dataframe):
        field_to_column_list = {}
        for columns_name, target_field in process_info.data_column_field_name_mapping.items():
            assert target_field in self.discrete_fields or target_field in self.continuous_fields
            if target_field not in field_to_column_list:
                field_to_column_list[target_field] = []
            field_to_column_list[target_field].append(columns_name)
        
        for field_name in self.discrete_fields:
            if field_name in field_to_column_list:
                columns = field_to_column_list[field_name]
                all_values = np.unique(np.concatenate([
                    dataframe[col_name].replace(process_info.data_column_default_value[col_name], np.nan).dropna().unique() \
                        if col_name in process_info.data_column_default_value \
                            else dataframe[col_name].dropna().unique()
                    for col_name in columns
                ]))
                
                for value in all_values:
                    self.discrete_collector.record_field_value(field_name, value)
        
        for field_name in self.continuous_fields:
            if field_name in field_to_column_list:
                columns = field_to_column_list[field_name]
                n_list = []
                mean_list = []
                std_list = []
                for col_name in columns:
                    serise = dataframe[col_name]\
                        .replace(process_info.data_column_default_value[col_name], np.nan)\
                            .dropna()\
                                if col_name in process_info.data_column_default_value \
                                    else dataframe[col_name].dropna()
                    n_list.append(serise.count())
                    mean_list.append(serise.mean())
                    std_list.append(serise.std())
                self.continuous_mean_std[field_name] = _calculate_overall_mean_std(n_list, mean_list, std_list)
                self.continuous_field_order.append(field_name)
    
    def is_discrete(self, filed_name):
        if filed_name in self.discrete_fields:
            return True
        elif filed_name in self.continuous_fields:
            return False
        else:
            raise Exception(f"Unrecognized field name {filed_name}")


def get_preprocess_function(field_meta: FieldMeta, process_info: ProcessInfo) -> callable:
    def preprocess_one_data(dict_like_input):
        discrete_mask = []
        discrete_index = []
        continue_mask = []
        continue_index = []
        continue_value = []

        def handle_one_column(column):
            field_name = process_info.data_column_field_name_mapping[column]
            value = dict_like_input[column]
            if field_meta.is_discrete(field_name):
                if pd.isna(value) or (column in process_info.data_column_default_value \
                    and value==process_info.data_column_default_value[column]):
                    # for mask. last index for pad embedding
                    discrete_mask.append(0)
                    discrete_index.append(len(field_meta.discrete_collector))
                else:
                    discrete_mask.append(1)
                    discrete_index.append(field_meta.discrete_collector.get_embedding_idx(field_name, value))
            else:
                if pd.isna(value) or (column in process_info.data_column_default_value \
                    and value==process_info.data_column_default_value[column]):
                    continue_mask.append(0)
                    continue_value.append(0)
                    continue_index.append(len(field_meta.continuous_field_order))
                else:
                    continue_mask.append(1)
                    mean, std = field_meta.continuous_mean_std[field_name]
                    continue_value.append((value-mean)/std)
                    continue_index.append(field_meta.continuous_field_order.index(field_name))

        for element in process_info.data_column_agg:
            if isinstance(element, list):
                for column in element:
                    handle_one_column(column)
            else:
                handle_one_column(element)
        
        return {
            'discrete_index': torch.IntTensor(discrete_index),
            'discrete_mask': torch.IntTensor(discrete_mask),  
            'continue_value': torch.FloatTensor(continue_value), 
            'continue_index': torch.IntTensor(continue_index),
            'continue_mask': torch.IntTensor(continue_mask)
        }

    return preprocess_one_data


def get_batch_preprocess_function(field_meta: FieldMeta, process_info: ProcessInfo) -> callable:
    disc_meta = []   # (column, field_dict, default_value, pad_idx)
    cont_meta = []   # (column, mean, std, order_idx, default_value, pad_idx)
    for element in process_info.data_column_agg:
        columns = element if isinstance(element, list) else [element]
        for column in columns:
            field_name = process_info.data_column_field_name_mapping[column]
            if field_meta.is_discrete(field_name):
                disc_meta.append((
                    column,
                    field_meta.discrete_collector.field_dict.get(field_name, {}),
                    process_info.data_column_default_value.get(column),
                    len(field_meta.discrete_collector),
                ))
            else:
                mean, std = field_meta.continuous_mean_std[field_name]
                cont_meta.append((
                    column, mean, std,
                    field_meta.continuous_field_order.index(field_name),
                    process_info.data_column_default_value.get(column),
                    len(field_meta.continuous_field_order),
                ))

    disc_cols = [m[0] for m in disc_meta]
    disc_fmaps = [{float(k): v for k, v in m[1].items()} for m in disc_meta]
    disc_default = [m[2] for m in disc_meta]
    disc_pad = [m[3] for m in disc_meta]

    cont_cols = [m[0] for m in cont_meta]
    cont_mean = np.array([m[1] for m in cont_meta], dtype=np.float64)
    cont_std = np.array([m[2] for m in cont_meta], dtype=np.float64)
    cont_order = np.array([m[3] for m in cont_meta], dtype=np.int64)
    cont_default = [m[4] for m in cont_meta]
    cont_pad = cont_meta[0][5] if cont_meta else 0

    def preprocess_batch(pl_df):
        n = pl_df.height

        if disc_cols:
            arr = pl_df.select(disc_cols).to_numpy()  # (n, k) float64
            isnull = np.isnan(arr)
            for i, default in enumerate(disc_default):
                if default is not None:
                    isnull[:, i] = isnull[:, i] | (arr[:, i] == default)
            idx = np.full((n, len(disc_cols)), 0, dtype=np.int64)
            for i, fm in enumerate(disc_fmaps):
                idx[:, i] = disc_pad[i]
                for fval, fidx in fm.items():
                    idx[arr[:, i] == fval, i] = fidx
            disc_idx, disc_mask = idx, (~isnull).astype(np.int64)
        else:
            disc_idx = np.empty((n, 0), dtype=np.int64)
            disc_mask = np.empty((n, 0), dtype=np.int64)

        if cont_cols:
            arr = pl_df.select(cont_cols).to_numpy()
            isnull = np.isnan(arr)
            for i, default in enumerate(cont_default):
                if default is not None:
                    isnull[:, i] = isnull[:, i] | (arr[:, i] == default)
            val = (arr - cont_mean) / cont_std
            val[isnull] = 0.0
            cidx = np.where(isnull, cont_pad, cont_order).astype(np.int64)
            con_val, con_idx, con_mask = val, cidx, (~isnull).astype(np.int64)
        else:
            con_val = np.empty((n, 0), dtype=np.float64)
            con_idx = np.empty((n, 0), dtype=np.int64)
            con_mask = np.empty((n, 0), dtype=np.int64)

        return (
            torch.from_numpy(disc_idx).int(),
            torch.from_numpy(disc_mask).int(),
            torch.from_numpy(con_val).float(),
            torch.from_numpy(con_idx).int(),
            torch.from_numpy(con_mask).int(),
        )

    return preprocess_batch

class EmbeddingModule(torch.nn.Module):

    def __init__(
        self, 
        dim, 
        bin_num,
        discrete_embedding_num, 
        continuous_num, 
        transform_matrix: np.ndarray
    ) -> None:
        super().__init__()
        self.dim, self.bin_num = dim, bin_num
        self.discrete_embedding = torch.nn.Embedding(discrete_embedding_num + 1, dim, discrete_embedding_num)
        self.continuous_v_embedding = torch.nn.Embedding(continuous_num + 1, bin_num, continuous_num)
        self.continuous_b1_embedding = torch.nn.Embedding(continuous_num + 1, bin_num, continuous_num)
        self.continuous_w_embedding = torch.nn.Embedding(continuous_num + 1, dim*bin_num, continuous_num)
        self.continuous_b2_embedding = torch.nn.Embedding(continuous_num + 1, dim, continuous_num)
        self.norm = torch.nn.LayerNorm(bin_num)
        self.act = torch.nn.ReLU()
        self.register_buffer("transform_matrix", torch.tensor(transform_matrix, dtype=torch.float32))

    def forward(
        self, 
        discrete_index,
        discrete_mask,  
        continue_value, 
        continue_index,
        continue_mask
    ):
        discrete_part = self.discrete_embedding(discrete_index)
        v = self.continuous_v_embedding(continue_index)
        b1 = self.continuous_b1_embedding(continue_index)
        w = self.continuous_w_embedding(continue_index)
        batch, leng = w.size()[0:2]
        b2 = self.continuous_b2_embedding(continue_index)
        continuous_part = (self.act(self.norm(continue_value.unsqueeze(-1) * v + b1)).unsqueeze(2) @ w.view(batch, leng, self.bin_num, self.dim)).squeeze() + b2

        with torch.no_grad():
            full_mask = (1 - torch.sign(self.transform_matrix @ torch.concat([1-discrete_mask, 1-continue_mask], dim=1).unsqueeze(-1).float())).int()

        return self.transform_matrix @ torch.concat([discrete_part, continuous_part], dim=1) , full_mask.squeeze()

def _get_embedding_module_para(field_meta: FieldMeta, process_info: ProcessInfo) -> dict:
    
    all_num = 0
    discrete_num = 0
    continuous_num = 0
    for element in process_info.data_column_agg:
        if isinstance(element, list):
            for column in element:
                all_num+=1
                field_name = process_info.data_column_field_name_mapping[column]
                if field_meta.is_discrete(field_name):
                    discrete_num+=1
                else:
                    continuous_num+=1
        else:
            all_num += 1
            field_name = process_info.data_column_field_name_mapping[element]
            if field_meta.is_discrete(field_name):
                discrete_num+=1
            else:
                continuous_num+=1

    transform_matrix = np.zeros([len(process_info.data_column_agg), all_num])
    j = 0
    i_d = 0
    i_c = 0
    for element in process_info.data_column_agg:
        if isinstance(element, list):
            for column in element:
                field_name = process_info.data_column_field_name_mapping[column]
                if field_meta.is_discrete(field_name):
                    transform_matrix[j, i_d] = 1
                    i_d+=1
                else:
                    transform_matrix[j, i_c+discrete_num] = 1
                    i_c+=1
        else:
            field_name = process_info.data_column_field_name_mapping[element]
            if field_meta.is_discrete(field_name):
                transform_matrix[j, i_d] = 1
                i_d+=1
            else:
                transform_matrix[j, i_c+discrete_num] = 1
                i_c+=1
        j+=1
    
    return {
        "discrete_embedding_num": len(field_meta.discrete_collector), 
        "continuous_num": len(field_meta.continuous_field_order), 
        "transform_matrix":transform_matrix
    }


def get_embedding_module(field_meta: FieldMeta, process_info: ProcessInfo, dim, bin_num) -> torch.nn.Module:
    return EmbeddingModule(dim, bin_num, **_get_embedding_module_para(field_meta, process_info))


def get_meta_process_info_from_dataframe(dataframe):
    pi = ProcessInfo()
    pi.data_column_field_name_mapping = {}
    pi.data_column_default_value = {}
    pi.data_column_agg = []
    discrete_columns = []
    continuous_columns = []
    nsamples = dataframe.shape[0]
    for column in dataframe.columns:
        count = len(dataframe[column].unique())
        if count > 100:
            continuous_columns.append(column)
            pi.data_column_field_name_mapping[column] = column
            pi.data_column_agg.append(column)
        else:
            discrete_columns.append(column)
            pi.data_column_field_name_mapping[column] = column
            pi.data_column_agg.append(column)
        vc = dataframe[column].value_counts()
        top_value = vc.index[0]
        top_count = vc.iloc[0]
        if top_count>0.9*nsamples:
            pi.data_column_default_value[column] = top_value
    return FieldMeta(discrete_columns, continuous_columns, pi, dataframe), pi


def save(path, fm: FieldMeta, pi: ProcessInfo):
    json_dict = {
        "field_meta": {
            "discrete_fields": fm.discrete_fields,
            "field_dict": fm.discrete_collector.field_dict,
            "embedding_num": fm.discrete_collector.embedding_num,
            "continuous_field_order": fm.continuous_field_order,
            "continuous_fields": fm.continuous_fields,
            "continuous_mean_std": fm.continuous_mean_std
        },
        "process_info": {
            "data_column_agg": pi.data_column_agg,
            "data_column_default_value": pi.data_column_default_value,
            "data_column_field_name_mapping": pi.data_column_field_name_mapping
        }
    }
    with open(path, 'w') as w:
        json.dump(json_dict, w)


def load(path):
    with open(path, 'r') as r:
        json_dict = json.load(r)
    pi = ProcessInfo()
    pi.data_column_agg = json_dict['process_info']['data_column_agg']
    pi.data_column_default_value = json_dict['process_info']['data_column_default_value']
    pi.data_column_field_name_mapping = json_dict['process_info']['data_column_field_name_mapping']
    fm = FieldMeta(None, None, None, None, False)
    fm.continuous_field_order = json_dict['field_meta']['continuous_field_order']
    fm.discrete_fields = json_dict['field_meta']['discrete_fields']
    fm.continuous_fields = json_dict['field_meta']['continuous_fields']
    fm.continuous_mean_std = json_dict['field_meta']['continuous_mean_std']
    fm.discrete_collector.embedding_num = json_dict['field_meta']['embedding_num']
    fm.discrete_collector.field_dict = json_dict['field_meta']['field_dict']
    return fm, pi