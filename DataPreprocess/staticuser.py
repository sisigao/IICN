import pandas as pd
import matplotlib.pyplot as plt

# 读取 SearchStream 数据（包含用户点击信息）
search_stream = pd.read_csv('SearchStream.tsv', sep='\t')

# 读取 SearchInfo 数据（如果需要用户相关的信息）
# search_info = pd.read_csv('SearchInfo.tsv', sep='\t')

# 筛选点击记录
clicks = search_stream[search_stream['IsClick'] == 1]

# 统计每个用户的点击次数
user_click_counts = clicks.groupby('UserID')['IsClick'].count().reset_index()
user_click_counts.columns = ['UserID', 'ClickCount']

# 定义点击次数区间
bins = [0, 5, 10, 15, 20, 25, float('inf')]
labels = ['[1-5]', '[6-10]', '[11-15]', '[16-20]', '[21-25]', '[26+]']

# 将点击次数分组到指定的区间
user_click_counts['ClickRange'] = pd.cut(user_click_counts['ClickCount'], bins=bins, labels=labels, right=False)

# 统计每个区间内的用户数量
click_range_counts = user_click_counts['ClickRange'].value_counts().sort_index().reset_index()
click_range_counts.columns = ['ClickRange', 'UserCount']

# 将数据转换为适合绘制的形式
plot_data = click_range_counts.set_index('ClickRange')['UserCount']

# 绘制图表
plt.figure(figsize=(10, 6))
plot_data.plot(kind='bar', color='skyblue', edgecolor='black')

# 添加标题和标签
plt.title('用户点击数量分布图')
plt.xlabel('点击次数区间')
plt.ylabel('用户数量')

# 添加数值标签
for i, v in enumerate(plot_data):
    plt.text(i, v + 10, str(v), ha='center', va='bottom')

# 显示图表
plt.tight_layout()
plt.show()