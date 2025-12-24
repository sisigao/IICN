import pandas as pd, numpy as np 
import pickle, json, ast, re, os, string
import tqdm
import math, random
from collections import defaultdict
from collections import deque

from tqdm.contrib.concurrent import process_map

data_dir = './Data/'
raw_data_dir = data_dir + 'raw/'
new_data_dir = data_dir + 'new/'
sample_data_dir = data_dir + 'data/'
random.seed(10)
training_data_type = 'Train'  # 当前处理的是训练数据

flatten = lambda forest: [leaf for tree in forest for leaf in tree]

TestSearchInfo = pd.read_csv(new_data_dir + 'TestSearchInfo.csv')
ValSearchInfo = pd.read_csv(new_data_dir + 'ValSearchInfo.csv')
TrainSearchInfo = pd.read_csv(new_data_dir + 'TrainSearchInfo.csv')
TrainSearchInfo = TrainSearchInfo.sort_values(by="SearchDate")
ValSearchInfo = ValSearchInfo.sort_values(by="SearchDate")
TestSearchInfo = TestSearchInfo.sort_values(by="SearchDate")

AdsInfo = pd.read_csv(new_data_dir + 'AdsInfo_new.csv')
AdsInfo = AdsInfo.set_index('AdID')
SearchStream = pd.read_csv(new_data_dir + 'SearchStream_new.csv')
SearchStream = SearchStream.astype("int")

data_path = raw_data_dir + 'UserInfo.tsv'
UserInfo = pd.read_csv(data_path, sep="\t")
UserInfo = UserInfo.set_index('UserID')
Category = pd.read_csv(new_data_dir + 'Category_new.csv')
Category = Category.set_index('CategoryID')
Location = pd.read_csv(new_data_dir + 'Location_new.csv')
Location = Location.set_index('LocationID')
UserInfo = UserInfo.fillna(0).astype("int")
Category = Category.fillna(0).astype("int")
Location = Location.fillna(0).astype("int")
training_SearchInfo = eval(training_data_type+'SearchInfo')
# 使用 eval 函数动态地将 training_data_type（值为 'Train'）与 'SearchInfo' 拼接成字符串 'TrainSearchInfo'，




# #### 2. make search_stream_dict

SearchStream = SearchStream.sort_values(by='IsClick')
# 将 SearchStream 数据按 IsClick 列排序，目的是将点击记录（IsClick=1）集中在一起，方便后续处理。
click_search_ids = SearchStream[SearchStream['IsClick']==1]['SearchID'].unique().tolist()
# 筛选出所有被点击的记录（IsClick=1）。
#
# 提取这些记录中的 SearchID 并去重，生成唯一ID列表。
click_dict = dict([(y,True) for x,y in enumerate(sorted(set(click_search_ids)))])
# 作用：生成一个字典，键是点击过的 SearchID，值为 True。
#
# 示例：{1001: True, 1002: True, ...}

search_stream_dict_path = new_data_dir+'search_stream_dict.pickle'

if not os.path.exists(search_stream_dict_path):  # 生成字典的逻辑
    tqdm.tqdm.pandas()
    search_stream_dict = SearchStream.groupby('SearchID')['AdID'].progress_apply(list).to_dict()

    with open(search_stream_dict_path, 'wb') as f:
         pickle.dump(search_stream_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
else:  # 加载已有字典
    f = open(search_stream_dict_path, 'rb')
    search_stream_dict = pickle.load(f)


# #### 3. make user seq lists

num_ado_feature = 18  # 广告特征维度
num_query_feature = 16  # 搜索特征维度

SearchStream = SearchStream.set_index('SearchID').sort_index()
#  将 SearchID 设为索引并排序，加快按 SearchID 查找的速度。
def deal_with_user_seq(tuples_input):  # 定义处理用户序列的函数
    u_id = tuples_input[0]  # 用户ID
    search_id_list = tuples_input[1]  # 该用户的所有搜索ID列表
    drop_search_id = 0  # 记录无效的SearchID数量
    ad_seq_list = []  # 存储广告序列
    user_features = [u_id]  # 用户特征（此处仅为用户ID）

    new_search_id_list = []
    for search_id in search_id_list:
        if search_id not in search_stream_dict:
            drop_search_id += 1  # 统计无效的SearchID
            continue
        new_search_id_list.append(search_id)  # 保留有效的SearchID
    search_id_list = new_search_id_list  # 更新为有效列表

    search_id_list = list(reversed(search_id_list))  # 反转列表，从最近的搜索开始处理
    tmp_cnt = 6  # 最多保留6次搜索（点击后重置计数）
    for search_id in search_id_list:
        if tmp_cnt <= 0:
            continue  # 超过6次则跳过
        
        if search_id in click_dict:  # 如果该搜索有点击，重置计数器为6
            tmp_cnt = 6

        # 获取该搜索展示的广告列表
        ad_id_list = search_stream_dict[search_id]
        query_features = [search_id]  # 搜索特征（此处仅为SearchID）
        search_ads_list = [query_features, ad_id_list]  # 组合搜索和广告信息
        ad_seq_list.append(search_ads_list)
        tmp_cnt -= 1
        
    ad_seq_list = list(reversed(ad_seq_list))  # 恢复时间顺序（从旧到新）
    # 组合用户特征和广告序列
    user_seq_list = [user_features, ad_seq_list]
    return (user_seq_list, drop_search_id)

def get_user_seq(data_type):  #定义生成用户序列的函数
    drop_search_id = 0  # 总无效SearchID数量
    tmp_SearchInfo = eval(data_type+'SearchInfo')  # 动态加载数据集（如TrainSearchInfo
    data_search_info_dict_path = new_data_dir+data_type+'_search_info_dict.pickle'  # 处理用户搜索记录字典
    if not os.path.exists(data_search_info_dict_path):
        print(tmp_SearchInfo.shape) #(95,980,006, 10) (3043334, 10) (1616503, 10)
        # 按用户分组，收集每个用户的SearchID列表
        simple_SearchInfo = tmp_SearchInfo[['UserID', 'SearchID']]
        simple_SearchInfo_dict = simple_SearchInfo.groupby('UserID')['SearchID'].apply(list).to_dict()
        # 保存字典
        with open(data_search_info_dict_path, 'wb') as f:
            pickle.dump(simple_SearchInfo_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        # 直接加载已有字典
        f = open(data_search_info_dict_path, 'rb')
        simple_SearchInfo_dict = pickle.load(f)

    # 获取所有用户ID
    uids = tmp_SearchInfo['UserID'].unique().tolist()
    # 处理用户序列列表
    data_user_seq_lists_path = new_data_dir+data_type+'_user_seq_lists.pickle'
    if not os.path.exists(data_user_seq_lists_path):
        print(data_type, 'produce user seq')
        tmp_input = [(uid, simple_SearchInfo_dict[uid]) for uid in uids]
        Results = [deal_with_user_seq(x) for x in tmp_input]
        # Results = process_map(deal_with_user_seq, tmp_input, max_workers=num_workers, chunksize=10)
        user_seq_lists = [x[0] for x in Results] 
        drop_search_id = sum([x[1] for x in Results])
        with open(data_user_seq_lists_path, 'wb') as f:
            pickle.dump(user_seq_lists, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        f = open(data_user_seq_lists_path, 'rb')
        user_seq_lists = pickle.load(f)

    print(data_path, " drop search_ids: ", drop_search_id)
    # train: 3,438,855
    # val: 514,911 
    # test: 341,711
    return user_seq_lists
    """
    user_seq_list.pickle:
    user_seq = [uid], [ [page1], [page2], ...[pageN] ] 
    pageN = [sid], [uc_id, uc_id, uc_id, uc_id, c_id]
    """

# #### 4. produce sample

def get_query_feature(search_id, search_info):  # 获取搜索页面的特征
    # 提取基础搜索信息：搜索ID、IP、登录状态、时间戳
    query_features = [search_id] + search_info[['IPID', 'IsUserLoggedOn', 'TimeStamp']].loc[search_id].astype("int").values.tolist()
    # 'LocationID', 'CategoryID'
    # 提取地理位置信息：位置ID及其层级关系
    search_LocationID = search_info['LocationID'].loc[search_id]
    # 提取分类信息：类别ID及其层级关系
    search_CategoryID = search_info['CategoryID'].loc[search_id]
    search_location_feature = [search_LocationID]+Location[['Level', 'RegionID', 'CityID']].loc[search_LocationID].values.tolist()
    search_category_feature = [search_CategoryID]+Category[['Level', 'ParentCategoryID', 'SubcategoryID']].loc[search_CategoryID].values.tolist()
    # 提取搜索词和参数（转换为列表）
    query_words = search_info['SearchQuery'].loc[search_id]
    query_words = eval(query_words)
    query_params = search_info['SearchParams'].loc[search_id]
    query_params = eval(query_params)
    # 合并所有特征为一个列表（共16个特征）
    query_features = query_features + search_location_feature + search_category_feature + query_words + query_params
    # 'SearchID', 'IPID', 'IsUserLoggedOn', 'TimeStamp', 'LocationID', 'CategoryID', 'Query', 'Params'
    # 4 + 4 + 4 + 1 + 3 = 16
    assert len(query_features)==num_query_feature,'query_features: '+str(len(query_features)) 
    return query_features

def get_ad_features(search_stream, ad_id):  # 获取广告的特征，将一个广告的信息（如“耐克运动鞋广告”）转换为18个数字组成的特征列表。
    ad_CategoryID = AdsInfo['CategoryID'].loc[ad_id]
    ad_category_feature = [ad_CategoryID]+Category[['Level', 'ParentCategoryID', 'SubcategoryID']].loc[ad_CategoryID].values.tolist()
    ad_title = AdsInfo['Title'].loc[ad_id]  # 提取广告标题和参数（转换为列表）
    ad_title = eval(ad_title)
    ad_params = AdsInfo['Params'].loc[ad_id]
    ad_params = eval(ad_params)
    ad_category_feature
    # 合并广告特征
    ad_features = [ad_id] + ad_category_feature + ad_title + ad_params
    # 添加广告在搜素结果中的位置，历史点击率，是否被点击
    search_stream_features = search_stream[['Position', 'HistCTR', 'IsClick']].loc[ad_id].values.tolist()
    ad_features = ad_features + search_stream_features
    ## ad_id, position, ad_title_keyword, (ad_cate_id, ad_cate_level, ad_parent_cate_id, ad_sub_cate_id), hist_ctr_bin, ad_params
    # 15 + 2 + 1 = 18
    assert len(ad_features)==num_ado_feature,'ado_features: '+str(len(ad_features)) 
    return ad_features

def get_page_features(page, search_info):  # 获取整个搜索页面的特征
    # 将一个搜索页面（如“用户一次搜索展示的5个广告”）
    # 转换为170个数字（5广告×34特征）的特征列表，
    # 末尾添加广告数量。
    search_id = page[0][0]     # 获取搜索ID
    ad_id_list = page[1]      # 获取该搜索展示的广告列表
    # 获取搜索页面的特征
    query_features = get_query_feature(search_id, search_info)
    # 准备广告数据
    search_stream = SearchStream.loc[[search_id]]
    search_stream = search_stream.set_index('AdID').sort_index()
    # 处理每个广告的特征（最多保留5个广告）
    page_ads_features = []
    cut_len = min(len(ad_id_list), 5)
    ad_id_list = ad_id_list[:cut_len]
    for ad_id in ad_id_list:
        ad_features = get_ad_features(search_stream, ad_id)
        current_ad_feature = query_features + ad_features
        # 16 + 18 = 34
        # assert len(current_ad_feature)==34,'ads_features: '+str(len(current_ad_feature)) 
        page_ads_features.append(current_ad_feature)
    # 不足5个广告则用0填充
    for _ in range(5-len(ad_id_list)):
        page_ads_features.append(mask_ad_features)
    return flatten(page_ads_features) + [len(ad_id_list)]



sample_page_num = 5  # 5个页面
page_ad_num = 5  # 每个页面5个广告

num_user_features = 5  # 用户特征
num_ad_features = num_query_feature + num_ado_feature  # =34，查询特征+广告特征
mask_ad_features = [0]*num_ad_features  # 填充0
mask_page_features = mask_ad_features*page_ad_num+[0]
num_sample_features = 1 + num_user_features + (num_ad_features-1) + page_ad_num*len(mask_page_features) + 1
num_drop_user = 0
# training_sample_list = []
# recent_histroy = defaultdict(list)


def deal_with_train_seq_list(user_seq_list):  # 处理用户的完整行为序列，输入是用户完整的行为序列列表
    tmp_samples = []  # 初始化一个空列表，用于存储生成的训练样本
    tmp_recent = defaultdict(list)  # 初始化一个空字典，用于存储最近访问的页面
    u_id = user_seq_list[0][0]
    if u_id in UserInfo.index:
        user_features = [u_id]+UserInfo.loc[u_id].values.tolist()
    else:
        user_features = [u_id]+[0]*(num_user_features-1)
    # assert len(user_features)==5, 'user_features: '+str(len(user_features)) 
    ## UserID UserAgentID UserAgentOSID UserDeviceID UserAgentFamilyID

    page_seq = user_seq_list[1]

    search_info = training_SearchInfo.loc[[u_id]]  # 从数据中筛选出特定用户的所有搜索信息
    search_info = search_info.set_index('SearchID').sort_index()  # 将这些信息按searchid进行索引和排序
    
    begin_index = min(sample_page_num, len(page_seq)-1)
    # begin_index = min(5, len(page_seq))
    history_pages_deque = deque([])  # 初始化双端队列history_pages_deque存储历史页面特征
    begin_pages = page_seq[:begin_index]
    # begin_pages = page_seq[-begin_index:]
    # assert len(begin_pages)<=5, "wrong length"

    for history_page in begin_pages:
        page_ads_features = get_page_features(history_page, search_info)  # 提取每个页面的特征
        history_pages_deque.append(page_ads_features)  # 加入历史队列
    res_page_seq = page_seq[begin_index:]  # 剩余待处理的页面
    
    for i in range(len(res_page_seq)):
        page = res_page_seq[i]
        search_id = page[0][0]
        ad_id_list = page[1]

        search_stream = SearchStream.loc[[search_id]]  # 提取与搜索id对应的广告数据
        search_stream = search_stream.set_index('AdID').sort_index()  # 设置AD_ID为索引排序
        current_query_features = get_query_feature(search_id, search_info)  # 获取当前查询的特征

        history_pages_features = list(history_pages_deque)  # 将历史页面队列转为列表
        num_mask_pages = sample_page_num-len(history_pages_features)  # 计算需要填充的页面数量
        for _ in range(num_mask_pages):
            history_pages_features.append(mask_page_features)  # 用mask_page_features填充
        history_pages_features = flatten(history_pages_features) + [5-num_mask_pages]  # 展开特征列表并添加有效页面数量

        page_ads_features = []  #
        # cut_len = min(len(ad_id_list), 5)
        # ad_id_list = random.shuffle(ad_id_list)
        # ad_id_list = ad_id_list[:cut_len]
        # 判断是否为最后一个页面或有点击记录的页面，作为采样页面
        is_sample_page = False
        if i == len(res_page_seq)-1:  # 最后一页
            is_sample_page = True  # 标记为sample page
        if search_id in click_dict:  # 如果是被点击过的searchid
            is_sample_page = True  # 标记为sample page

        have_neg = False
        for ad_id in ad_id_list:
            current_ad_features = get_ad_features(search_stream, ad_id)  # 提取广告特征
            current_qad_feature = current_query_features + current_ad_features  # 查询特征与广告特征合并
            # assert len(current_qad_feature)==33,'ads_features: '+str(len(current_ad_feature)) 
            page_ads_features.append(current_qad_feature)
            if not is_sample_page: # not click page or not last page
                continue
            if search_stream['ObjectType'].loc[ad_id] != 3: # remove nan ad
                continue
            
            label = current_ad_features[-1]  # 从广告的特征列表中获取最后一个元素作为标签即是否被点击
            if label !=1:  # 如果未被点击
                if not have_neg and label == 0:  # 如果 还没有记录过未点击的广告 且 当前广告未被点击
                    have_neg = True  # 记录这个未点击的广告
                else:  # 如果已经记录过未点击的广告，则跳过当前广告，不进行下一步处理，
                    continue

            target_ad_features = current_ad_features[:-1] # 从开头到倒数第二个元素
            target_qad_feature = current_query_features + target_ad_features
            target_uqad_features = user_features + target_qad_feature 
            sample = [label] + target_uqad_features + history_pages_features
            assert len(sample) == num_sample_features, "sample length wrong!!! with len: "+str(len(sample))
            # training_sample_list.append(sample)
            tmp_samples.append(sample)

        for _ in range(page_ad_num-len(ad_id_list)):  # 对不足最大广告数的页面进行掩码填充
            page_ads_features.append(mask_ad_features)
        page_ads_features = flatten(page_ads_features) + [len(ad_id_list)]

        if len(history_pages_deque) >= sample_page_num:
            history_pages_deque.popleft()  # 从双端队列中移除最左侧也就是最早添加的元素
        history_pages_deque.append(page_ads_features)

    history_pages_features = list(history_pages_deque)
    num_mask_pages = sample_page_num-len(history_pages_features)
    for _ in range(num_mask_pages):
        history_pages_features.append(mask_page_features)
    history_features = flatten(history_pages_features) + [sample_page_num-num_mask_pages]
    # recent_histroy[u_id] = history_features
    tmp_recent[u_id] = history_features
    return (tmp_samples, tmp_recent)

#### 5. 主流程：生成并保存样本
train_sample_data_path = sample_data_dir+training_data_type+'_data.csv'
if not os.path.exists(train_sample_data_path):  # 如果训练样本不存在
    print('produce train sample')
    training_user_seq_lists = get_user_seq(training_data_type)  # 获取用户历史行为序列
    # 设置用户ID为索引
    training_SearchInfo = training_SearchInfo.set_index('UserID').sort_index()
    # 过滤无效用户
    new_training_user_seq_lists = []  # 初始化有效用户序列列表
    for user_seq_list in tqdm.tqdm(training_user_seq_lists, desc='filter train seq lists'):
        page_seq = user_seq_list[1]
        if len(page_seq) < 2:  # 如果搜索次数少于2次
            num_drop_user += 1  # 记录被丢弃的用户数
            continue  # 跳过该用户
        new_training_user_seq_lists.append(user_seq_list)  # 保留有效用户
    training_user_seq_lists = new_training_user_seq_lists  # 更新过滤后的列表

    # pool = Pool(10)
    num_workers = 20  # 使用20个CPU核心并行处理
    Results = process_map(  # 使用对进程并行处理函数
        deal_with_train_seq_list,   # 处理单个用户的函数
        training_user_seq_lists,  # 所有用户的序列数据
        max_workers=num_workers,  # 最大进程数
        chunksize=1)  # 每个进程1次处理1个用户
    # 收集生成的样本
    training_sample_list = []
    for x in tqdm.tqdm(Results, desc='update training sample list'):
        for y in x[0]:  # x[0]是单个用户的所有样本列表
            training_sample_list.append(y)  # 将样本添加到总列表
    # 保存用户的历史特征
    recent_histroy = dict()  # 初始化历史记录字典
    for x in tqdm.tqdm(Results, desc='update recent history dict'):
        recent_histroy.update(x[1]) # x[1]是用户的历史特征

    training_sample_list = np.array(training_sample_list)  # 转为NumPy数组
    training_data = pd.DataFrame(training_sample_list)     # 转为Pandas表格
    print('drop_user: ', num_drop_user)  # 打印被丢弃的用户数
    training_data.to_csv(sample_data_dir+training_data_type+'_data.csv', index=False)

if not os.path.exists(sample_data_dir+'recent_histroy.pickle'):
    with open(sample_data_dir+'recent_history.pickle', 'wb') as f:
        pickle.dump(recent_histroy, f)  # 将recent_histroy 对象序列化后保存到文件中。
else:
    f = open(sample_data_dir+'recent_history.pickle', 'rb')
    recent_histroy = pickle.load(f)


def deal_with_validate_seq_list(user_seq_list):
    # 处理验证集或测试集的单个用户行为序列，生成样本并更新历史记录
    tmp_samples = []    # 存储当前用户生成的样本（如广告是否被点击的样本）
    tmp_recent = defaultdict(list)  # 存储更新后的用户历史记录
    u_id = user_seq_list[0][0] # 用户ID（如用户1001）
    user_features = [u_id]+UserInfo.loc[u_id].values.tolist()  # 用户特征（设备、操作系统等）
    search_info = validate_SearchInfo.loc[[u_id]]  # 获取该用户的搜索记录
    search_info = search_info.set_index('SearchID')  # 按搜索ID排序

    recent_histroy_features = recent_histroy[u_id]
    history_features, history_page_lens = recent_histroy_features[:-1], recent_histroy_features[-1]
    history_features = np.array(history_features).reshape(-1, len(mask_page_features))

    history_features = history_features[:history_page_lens]  # 截取有效历史记录
    history_pages_deque = deque(list(history_features))  # 转为队列方便更新

    page_seq = user_seq_list[1]  # 用户的所有搜索记录列表  # page 的格式为 [[搜索ID], [广告1, 广告2, ...]]。
    for i in range(len(page_seq)):
        page = page_seq[i]
        search_id = page[0][0]
        ad_id_list = page[1]
        current_query_features = get_query_feature(search_id, search_info)
        search_stream = SearchStream.loc[[search_id]]
        search_stream = search_stream.set_index('AdID').sort_index()

        history_pages_features = list(history_pages_deque)
        num_mask_pages = sample_page_num-len(history_pages_features)
        for _ in range(num_mask_pages):
            history_pages_features.append(mask_page_features)
        history_pages_features = flatten(history_pages_features) + [5-num_mask_pages]

        page_ads_features = []

        is_sample_page = False

        if i == len(page_seq)-1:
            is_sample_page = True
        if search_id in click_dict:
            is_sample_page = True


        have_neg = False # 标记是否已有一个负样本（未点击）
        for ad_id in ad_id_list:  # 遍历该搜索展示的广告
            current_ad_features = get_ad_features(search_stream, ad_id)
            current_qad_feature = current_query_features + current_ad_features
            page_ads_features.append(current_qad_feature)
            if not is_sample_page:
                continue
            if search_stream['ObjectType'].loc[ad_id] != 3:
                continue

            label = current_ad_features[-1]
            if label !=1:
                if not have_neg and label == 0:
                    have_neg = True
                else:
                    continue

            label = current_ad_features[-1]
            target_ad_features = current_ad_features[:-1]
            target_quad_features = user_features + current_query_features + target_ad_features 
            sample = [label] + target_quad_features + history_pages_features
            assert len(sample) == num_sample_features, "sample length wrong!!! with len: "+str(len(sample))
            tmp_samples.append(sample)
        
        for _ in range(page_ad_num-len(ad_id_list)):
            page_ads_features.append(mask_ad_features)
        page_ads_features =  flatten(page_ads_features) + [len(ad_id_list)]

        if len(history_pages_deque) >= sample_page_num:
            history_pages_deque.popleft()
        history_pages_deque.append(page_ads_features)

    history_pages_features = list(history_pages_deque)
    num_mask_pages = sample_page_num-len(history_pages_features)
    for _ in range(num_mask_pages):
        history_pages_features.append(mask_page_features)
    history_features = flatten(history_pages_features) + [sample_page_num-num_mask_pages]
    # recent_histroy[u_id] = history_features
    tmp_recent[u_id] = history_features
    return (tmp_samples, tmp_recent)

print('begin produce val/test sample')
for validate_data_type in ['Val', 'Test']:
    validate_SearchInfo = eval(validate_data_type+'SearchInfo')
    validate_sample_list = []
    validate_drop_user = 0

    validate_user_seq_lists = get_user_seq(validate_data_type)
    validate_SearchInfo = validate_SearchInfo.set_index('UserID').sort_index()

    new_validate_user_seq_lists = []
    for user_seq_list in tqdm.tqdm(validate_user_seq_lists, desc='filte val seq lists'):
        u_id = user_seq_list[0][0]
        if u_id not in recent_histroy.keys(): # remove cold-start user 
            validate_drop_user += 1
            continue  # 跳过该用户
        new_validate_user_seq_lists.append(user_seq_list)
    validate_user_seq_lists = new_validate_user_seq_lists

    num_workers = 20
    Results = process_map(deal_with_validate_seq_list, validate_user_seq_lists, max_workers=num_workers, chunksize=1)

    validate_sample_list = []
    for x in tqdm.tqdm(Results, desc='update '+validate_data_type+' sample list'):
        for y in x[0]:
            validate_sample_list.append(y)  # 合并所有进程生成的样本到validate_sample_list
    recent_histroy = dict()
    for x in tqdm.tqdm(Results, desc='update recent history dict'):
        recent_histroy.update(x[1])

    validate_sample_list = np.array(validate_sample_list)
    validate_data = pd.DataFrame(validate_sample_list)
    print(validate_data_type, ' data:', validate_data.shape)
    print(validate_data_type,' drop_user: ', validate_drop_user) 
    validate_data.to_csv(sample_data_dir+validate_data_type+'_data.csv', index=False)
