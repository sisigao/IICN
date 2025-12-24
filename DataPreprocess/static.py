import pandas as pd

data_dir = '../Data/'
raw_data_dir = data_dir + 'raw/'  # 原始数据目录


data_path = raw_data_dir + 'trainSearchStream.tsv'
SearchStream = pd.read_csv(data_path, sep="\t")
# 统计广告数
ad_count = SearchStream['AdID'].nunique()

data_path = raw_data_dir + 'SearchInfo.tsv'
search_info = pd.read_csv(data_path, sep="\t")

# 统计用户数（假设UserInfo.tsv中包含所有用户）
user_count = search_info['UserID'].nunique()

