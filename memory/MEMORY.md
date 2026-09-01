# 核心记忆

## 项目目标
序列模型学习跨市场（A股/港股/美股）通用日K线交易信号，从OCHLV数据中获利。

## 已定技术决策
- **存储**: DuckDB + Parquet，列式存储，训练/测试/回测均从磁盘读
- **数据范围**: 三个市场能收集多远收集多远（估计~100M行，6-10GB）
- **特征**: 日OCHLV + 换手率(turnover_rate)，换手率存小数
- **复权**: 前复权价格
- **跨市场通用**: 换手率代替成交量作为主要活跃度指标（消除股本差异）

## 数据库 Schema
```
stock_meta:    stock_code(PK) | stock_name | market(A/HK/US) | start_dt | end_dt
ochlv_daily:   dt | stock_code | open | high | low | close | volume | turnover_rate(decimal)
```

## 风险关注点
- 幸存者偏差：必须包含退市股
- 跨市场交易日历不统一，时序构造需对齐
- 时间切分防穿越（非随机切分）
- 美股数据源需另找（tushare不覆盖）
- 早期成交量为0的行处理策略待定

## 环境
- Python: `/root/anaconda3/envs/ldm/bin/python3.8`
- 禁止pip install，需依赖告知领导安装
