import yaml

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

tushare_token = cfg["tushare"]["token"]