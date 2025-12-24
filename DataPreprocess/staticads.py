import pandas as pd
import matplotlib.pyplot as plt

# 定义数据目录
raw_data_dir = 'raw/'  # 原始数据目录

# 分块读取数据并统计每个用户的点击次数
chunk_size = 100000  # 每次读取100,000行

# 定义数据类型（不再包含IsClick的类型指定）
search_info_dtypes = {
    'SearchID': 'int32',
    'UserID': 'int32'
}

search_stream_dtypes = {
    'SearchID': 'int32',
    'AdID': 'int32'  # 移除了IsClick的类型声明
}

# 只读取必要的列
search_info_usecols = ['SearchID', 'UserID']
search_stream_usecols = ['SearchID', 'AdID', 'IsClick']

# 读取 SearchInfo 数据
search_info = pd.read_csv(
    raw_data_dir + 'SearchInfo.tsv',
    sep='\t',
    dtype=search_info_dtypes,
    usecols=search_info_usecols
)

print("读取读取 SearchInfo 数据完成")
# 初始化用户点击统计
user_click_counts = pd.Series(dtype='int32')

# 分块读取 SearchStream 数据
for chunk in pd.read_csv(
        raw_data_dir + 'trainSearchStream.tsv',
        sep='\t',
        chunksize=chunk_size,
        dtype=search_stream_dtypes,
        usecols=search_stream_usecols,
        converters={
            # 转换器优先处理数据清洗
            'IsClick': lambda x: 0 if x.strip() in ('', '-1') else int(x)
        }
):
    # 显式转换数据类型（关键修改）
    chunk['IsClick'] = chunk['IsClick'].astype('int8')

    # 筛选点击记录
    chunk_clicks = chunk[chunk['IsClick'] == 1]

    if not chunk_clicks.empty:
        # 合并数据获取UserID
        merged_chunk = pd.merge(
            chunk_clicks,
            search_info,
            on='SearchID',
            how='left'
        )

        # 统计当前块点击量
        chunk_user_counts = merged_chunk.groupby('UserID')['IsClick'].count()

        # 累加统计（保持int32类型）
        user_click_counts = user_click_counts.add(
            chunk_user_counts,
            fill_value=0
        ).astype('int32')
print("读取读取 Searchstream数据完成且合并完成")

# 后续处理与可视化保持不变
user_click_counts = user_click_counts.reset_index()
user_click_counts.columns = ['UserID', 'ClickCount']

# 定义区间分组
bins = [0, 5, 10, 15, 20, 25, float('inf')]
labels = ['[1-5]', '[6-10]', '[11-15]', '[16-20]', '[21-25]', '[26+]']

user_click_counts['ClickRange'] = pd.cut(
    user_click_counts['ClickCount'],
    bins=bins,
    labels=labels,
    right=False
)

# 统计区间用户数
click_range_counts = user_click_counts['ClickRange'].value_counts().sort_index()
plot_data = click_range_counts.reset_index()
plot_data.columns = ['ClickRange', 'UserCount']

print("统计完成")

# 绘制图表
plt.figure(figsize=(10, 6))
ax = plot_data.set_index('ClickRange')['UserCount'].plot(
    kind='bar',
    color='skyblue',
    edgecolor='black'
)

plt.title('user', fontsize=14)
plt.xlabel('click', fontsize=12)
plt.ylabel('user-number', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yscale('log')

# 添加数值标签
for i, v in enumerate(plot_data['UserCount']):
    ax.text(i, v * 1.1, f'{v:,}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()
plt.savefig('plot.png')