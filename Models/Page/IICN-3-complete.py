import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from Models.utils.layer import Attention, MultiLayerPerceptron


class IICN(nn.Module):
    def __init__(self, Sampler, ModelSettings):
        super().__init__()

        # 初始化参数
        self.num_features_dict = Sampler.num_features_dict
        self.embed_dim = eval(ModelSettings['embed_dim'])
        dnn_dim_list = eval(ModelSettings['dnn_dim_list'])
        self.page_layer = ModelSettings['page_layer']
        self.remove_nan = eval(ModelSettings['remove_nan'])
        mha_head_num = eval(ModelSettings['mha_head_num'])

        # 构建嵌入层
        self._build_embedding_layer(self.num_features_dict)
        self.ad_embed_dim = self.cnt_fts_dict['ad_embed'] * self.embed_dim
        self.tad_embed_dim = (self.cnt_fts_dict['ad_embed'] - 2) * self.embed_dim
        self.qy_embed_dim = self.cnt_fts_dict['qy_embed'] * self.embed_dim
        self.adq_embed_dim = self.ad_embed_dim + self.qy_embed_dim

        # 新建参数：用于信息熵和时间权重计算
        self.alpha_dim = 128  # 相关度计算维度
        self.W_Q = nn.Linear(self.ad_embed_dim, self.alpha_dim, bias=False)  # 修正输入维度
        self.W_K = nn.Linear(self.ad_embed_dim, self.alpha_dim, bias=False)  # 修正输入维度
        self.W_V = nn.Linear(self.ad_embed_dim, self.ad_embed_dim, bias=False)  # 修正输入维度
        self.lambda_param = nn.Parameter(torch.tensor(0.1))  # 可学习的时间衰减参数
        self.gate_linear = nn.Linear(2, 1)  # 用于动态融合权重

        if self.page_layer == 'dynamic_page':
            alpha_input_dim = self.ad_embed_dim + self.tad_embed_dim
            alpha_dim_list = eval(ModelSettings['alpha_dim_list'])
            self.page_net = nn.ModuleDict({
                'gru': nn.GRU(self.ad_embed_dim, self.ad_embed_dim, num_layers=1),
                'target_to_adq': nn.Sequential(
                    nn.Linear(self.tad_embed_dim, self.ad_embed_dim), nn.ReLU()
                ),
                'din1': Attention(self.ad_embed_dim, ModelSettings),
                'alpha1': MultiLayerPerceptron(alpha_input_dim, alpha_dim_list, dropout=0, activation=nn.ReLU(),
                                               output_layer=True),
                'target_to_pq': nn.Sequential(
                    nn.Linear(self.tad_embed_dim, self.ad_embed_dim), nn.ReLU()
                ),
                'mha2': nn.MultiheadAttention(self.adq_embed_dim, num_heads=mha_head_num),
                'din2': Attention(self.ad_embed_dim, ModelSettings),
                'alpha2': MultiLayerPerceptron(alpha_input_dim, alpha_dim_list, dropout=0, activation=nn.ReLU(),
                                               output_layer=True),
            })
        else:
            raise ValueError('unknown PIN page layer name: ', self.page_layer)

        self.atten_net = torch.nn.MultiheadAttention(self.ad_embed_dim, num_heads=mha_head_num)

        # 更新DNN输入维度（包含用户兴趣向量）
        dnn_input_dim = self.cnt_fts_dict['user'] * self.embed_dim + self.tad_embed_dim + self.ad_embed_dim
        self.dnn_net = nn.ModuleDict({
            'dnn': MultiLayerPerceptron(dnn_input_dim, dnn_dim_list, dropout=0, activation=nn.PReLU(),
                                        output_layer=False)
        })

        # 添加融合MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(4 * self.ad_embed_dim, self.ad_embed_dim),
            nn.ReLU(),
            nn.Linear(self.ad_embed_dim, self.ad_embed_dim)
        )
        self.logits_linear = nn.Linear(dnn_dim_list[-1], 1)
        self.init_weights()

    def _build_embedding_layer(self, num_features_dict):
        self.embedding_dict = nn.ModuleDict()
        self.cnt_fts_dict = OrderedDict()

        # 单值特征嵌入
        for key_name in ['user', 'ad', 'location', 'category']:
            num_features_list = num_features_dict[key_name]
            if key_name == 'ad':
                num_features_list = num_features_list[1:]  # 删除search_id
            self.embedding_dict[key_name] = nn.ModuleList(nn.Embedding(x, self.embed_dim) for x in num_features_list)
            self.cnt_fts_dict[key_name] = len(num_features_list)

        # 多值特征嵌入
        for key_name in ['ad_title', 'ad_params', 'search_query', 'search_params'] + ['page_click_num']:
            self.embedding_dict[key_name] = nn.Embedding(num_features_dict[key_name], self.embed_dim)

        # 特征数量统计
        self.cnt_fts_dict['multi'] = sum(num_features_dict['multi'].values())
        self.cnt_fts_dict['ad_embed'] = self.cnt_fts_dict['ad'] + self.cnt_fts_dict['location'] + \
                                        self.cnt_fts_dict['category'] * 2 + len(
            num_features_dict['multi'].values()) + 1  # + page_click_num(1)
        self.cnt_fts_dict['qy_embed'] = 13 + 1

    def __feature_embedding(self, features, embedding_name):
        num_col = features.shape[1]
        features_embed_list = []
        for col in range(num_col):
            features_embed_list.append(self.embedding_dict[embedding_name][col](features[:, col]))
        return torch.cat(features_embed_list, 1)

    def __ad_embedding(self, ad_features, is_target=False):
        ad_embedding_dict = OrderedDict()
        loc_begin = 3
        ad_begin = 15
        fts_end = 33
        if is_target:
            fts_end -= 1  # 目标广告忽略is_click列
        begin_i = loc_begin

        # 提取时间戳（第2列）
        timestamps = ad_features[:, 2].clone() if ad_features.size(1) > 2 else None

        for key_name in ['location', 'category']:
            tmp_features = ad_features[:, begin_i:begin_i + self.cnt_fts_dict[key_name]]
            ad_embedding_dict[key_name] = self.__feature_embedding(tmp_features, key_name)
            begin_i += self.cnt_fts_dict[key_name]

        for key_name in ['search_query', 'search_params']:
            tmp_features = ad_features[:, begin_i:begin_i + self.num_features_dict['multi'][key_name]]
            ad_embedding_dict[key_name] = nn.functional.embedding(tmp_features, self.embedding_dict[key_name].weight)
            begin_i += self.num_features_dict['multi'][key_name]

        begin_i = ad_begin + 1  # ad_id
        key_name = 'category'
        tmp_features = ad_features[:, begin_i:begin_i + self.cnt_fts_dict[key_name]]
        ad_embedding_dict['ad_' + key_name] = self.__feature_embedding(tmp_features, key_name)
        begin_i += self.cnt_fts_dict[key_name]

        for key_name in ['ad_title', 'ad_params']:
            tmp_features = ad_features[:, begin_i:begin_i + self.num_features_dict['multi'][key_name]]
            ad_embedding_dict[key_name] = nn.functional.embedding(tmp_features, self.embedding_dict[key_name].weight)
            begin_i += self.num_features_dict['multi'][key_name]

        tmp_features = torch.cat((
            ad_features[:, :loc_begin],
            ad_features[:, ad_begin].view(-1, 1),
            ad_features[:, begin_i:fts_end]
        ), 1)
        ad_embedding_dict['uni'] = self.__feature_embedding(tmp_features, 'ad')

        ad_features = torch.cat((
            ad_embedding_dict['uni'],
            ad_embedding_dict['location'],
            ad_embedding_dict['category'],
            ad_embedding_dict['search_query'].view(-1, self.embed_dim),
            ad_embedding_dict['search_params'].sum(1).view(-1, self.embed_dim),
            ad_embedding_dict['ad_category'],
            ad_embedding_dict['ad_title'].sum(1).view(-1, self.embed_dim),
            ad_embedding_dict['ad_params'].sum(1).view(-1, self.embed_dim)
        ), 1)

        return ad_features, timestamps

    def __query_embedding(self, query_features):
        query_embedding_dict = OrderedDict()
        loc_begin = 3
        fts_end = 15
        begin_i = loc_begin

        for key_name in ['location', 'category']:
            tmp_features = query_features[:, begin_i:begin_i + self.cnt_fts_dict[key_name]]
            query_embedding_dict[key_name] = self.__feature_embedding(tmp_features, key_name)
            begin_i += self.cnt_fts_dict[key_name]

        for key_name in ['search_query', 'search_params']:
            tmp_features = query_features[:, begin_i:begin_i + self.num_features_dict['multi'][key_name]]
            query_embedding_dict[key_name] = nn.functional.embedding(tmp_features, self.embedding_dict[key_name].weight)
            begin_i += self.num_features_dict['multi'][key_name]

        tmp_features = query_features[:, :loc_begin]
        query_embedding_dict['uni'] = self.__feature_embedding(tmp_features, 'ad')

        query_features = torch.cat((
            query_embedding_dict['uni'],
            query_embedding_dict['location'],
            query_embedding_dict['category'],
            query_embedding_dict['search_query'].view(-1, self.embed_dim),
            query_embedding_dict['search_params'].sum(1).view(-1, self.embed_dim),
        ), 1)
        return query_features

    def _make_embedding_layer(self, features):
        embedding_layer = OrderedDict()
        batch_size = features.shape[0]
        embedding_layer['batch_size'] = batch_size

        cnt_user_fts = self.cnt_fts_dict['user']
        cnt_qad_fts = self.cnt_fts_dict['ad'] + self.cnt_fts_dict['location'] + \
                      self.cnt_fts_dict['category'] * 2 + self.cnt_fts_dict['multi']
        cnt_qpage_fts = cnt_qad_fts * 5 + 1 + 1
        fts_index_bias = cnt_user_fts + cnt_qad_fts - 1

        embedding_layer['user'] = self.__feature_embedding(features[:, :cnt_user_fts], 'user')
        target_embed, target_timestamps = self.__ad_embedding(features[:, cnt_user_fts:fts_index_bias], is_target=True)
        embedding_layer['target'] = target_embed
        embedding_layer['target_timestamps'] = target_timestamps

        page_size = 5
        ad_size = 5
        query_size = 15
        page_seq = []
        page_timestamps = []  # 存储每个广告的时间戳
        query_seq = []
        num_page_ads = []
        mask_nan_ad = torch.ones((batch_size, page_size, ad_size), device=features.device)
        mask_click_ad = torch.ones((batch_size, page_size, ad_size), device=features.device)

        for i in range(page_size):
            page_index_bias = fts_index_bias + i * cnt_qpage_fts
            ad_embed_seq = []
            ad_timestamp_seq = []
            for j in range(ad_size):
                ad_start = page_index_bias + j * cnt_qad_fts
                ad_end = page_index_bias + (j + 1) * cnt_qad_fts
                ad_embed, ad_timestamp = self.__ad_embedding(features[:, ad_start:ad_end])
                ad_embed_seq.append(ad_embed)
                ad_timestamp_seq.append(ad_timestamp)

            ad_embed_seq = torch.stack(ad_embed_seq, 1)  # [B, ad_size, D]
            ad_timestamp_seq = torch.stack(ad_timestamp_seq, 1)  # [B, ad_size]
            page_timestamps.append(ad_timestamp_seq)

            num_click = features[:, page_index_bias + ad_size * cnt_qad_fts + 1].view(-1)
            num_click_embed = self.embedding_dict['page_click_num'](num_click)
            num_click_a = num_click_embed.unsqueeze(1).repeat(1, ad_size, 1)
            ad_embed_seq = torch.cat((ad_embed_seq, num_click_a), dim=2)
            page_seq.append(ad_embed_seq)

            query_embed = self.__query_embedding(features[:, page_index_bias:page_index_bias + query_size])
            query_embed = torch.cat((query_embed, num_click_embed), dim=1)
            query_seq.append(query_embed)

            num_ad = features[:, page_index_bias + ad_size * cnt_qad_fts].view(-1)
            num_page_ads.append(num_ad)

            for j in range(ad_size):
                ad_features = features[:, page_index_bias + j * cnt_qad_fts:page_index_bias + (j + 1) * cnt_qad_fts]
                mask_nan_ad[:, i, j] = (ad_features[:, -1] == 2)
                mask_click_ad[:, i, j] = (ad_features[:, -1] == 1)

        embedding_layer['page_seq'] = torch.stack(page_seq, 1)
        embedding_layer['page_timestamps'] = torch.stack(page_timestamps, 1)  # [B, P, A]
        embedding_layer['query_seq'] = torch.stack(query_seq, 1)
        embedding_layer['num_page_ads'] = torch.stack(num_page_ads, 1).view(batch_size, page_size)
        embedding_layer['num_pages'] = features[:, -1].view(-1)
        embedding_layer['mask_nan_ad'] = mask_nan_ad.bool()
        embedding_layer['mask_click_ad'] = mask_click_ad.bool()

        return embedding_layer

    def _process_click_unclick(self, clicked_ads, unclicked_ads, clicked_ts):
        """
        处理点击和未点击物品，计算信息熵权重和融合表示
        clicked_ads: [num_clicks, ad_embed_dim]
        unclicked_ads: [num_unclicks, ad_embed_dim]
        clicked_ts: [num_clicks] 时间戳
        返回: (new_click_reps, wi_weights, timestamps)
        """
        if len(clicked_ads) == 0:
            return [], [], []

        # 归一化嵌入向量
        clicked_norm = F.normalize(clicked_ads, p=2, dim=1)
        unclicked_norm = F.normalize(unclicked_ads, p=2, dim=1) if unclicked_ads is not None and len(
            unclicked_ads) > 0 else None

        # 计算余弦相似度
        assignments = torch.tensor([], dtype=torch.long, device=clicked_ads.device)
        if unclicked_norm is not None:
            sim_matrix = torch.mm(clicked_norm, unclicked_norm.t())  # [num_clicks, num_unclicks]
            # 为每个未点击物品分配最相似的点击物品
            _, assignments = torch.max(sim_matrix, dim=0)  # [num_unclicks]

        new_click_reps = []
        wi_weights = []
        timestamps = []

        for j in range(len(clicked_ads)):
            # 找到分配给当前点击物品的未点击物品
            uk_indices = (assignments == j).nonzero(as_tuple=True)[0]
            class_unclicked = unclicked_ads[uk_indices] if len(uk_indices) > 0 else None

            # 计算相关度
            d_j = clicked_ads[j]
            q_j = self.W_Q(d_j)
            k_j = self.W_K(d_j)
            alpha_j = torch.dot(q_j, k_j)

            alpha_jk = []
            if class_unclicked is not None:
                K_uk = self.W_K(class_unclicked)  # [num_in_class, alpha_dim]
                alpha_jk = torch.mv(K_uk, q_j)  # [num_in_class]

            # 标准化相关度
            all_alpha = torch.cat([alpha_j.view(1), alpha_jk]) if len(alpha_jk) > 0 else alpha_j.view(1)
            beta = F.softmax(all_alpha, dim=0)
            beta_j = beta[0]
            beta_jk = beta[1:] if len(alpha_jk) > 0 else torch.tensor([], device=clicked_ads.device)

            # 计算信息熵权重
            entropy = 0.0
            if len(beta_jk) > 0:
                entropy = -torch.sum(beta_jk * torch.log2(beta_jk + 1e-9))
            w_j = entropy - beta_j * torch.log2(beta_j + 1e-9)

            # 融合未点击物品
            X_j = torch.zeros_like(d_j)
            if class_unclicked is not None and len(class_unclicked) > 0:
                mu_jk = F.softmax(alpha_jk, dim=0)
                V_uk = self.W_V(class_unclicked)  # [num_in_class, ad_embed_dim]
                X_j = torch.sum(mu_jk.unsqueeze(1) * V_uk, dim=0)

            # 创建新的点击物品表示
            concat_features = torch.cat([
                d_j,
                X_j,
                d_j - X_j,
                d_j * X_j
            ])

            # 使用MLP融合特征
            new_rep = self.fusion_mlp(concat_features)

            new_click_reps.append(new_rep)
            wi_weights.append(w_j)
            timestamps.append(clicked_ts[j])

        return new_click_reps, wi_weights, timestamps

    def _page_layer(self, embedding_layer):
        page_layer = OrderedDict()
        page_seq = embedding_layer['page_seq']  # [B, P, A, D]
        page_timestamps = embedding_layer['page_timestamps']  # [B, P, A]
        target = embedding_layer['target']  # [B, D]
        num_page_ads = embedding_layer['num_page_ads']  # [B, P]
        num_pages = embedding_layer['num_pages']  # [B]
        mask_nan_ad = embedding_layer['mask_nan_ad']  # [B, P, A]
        mask_click_ad = embedding_layer['mask_click_ad']  # [B, P, A]

        device = page_seq.device
        batch_size = page_seq.shape[0]
        page_size = page_seq.shape[1]
        ad_size = page_seq.shape[2]

        # 修复维度不匹配问题 - 使用广播机制创建广告掩码
        ad_indices = torch.arange(ad_size, device=device).view(1, 1, ad_size)
        num_page_ads_expanded = num_page_ads.unsqueeze(-1)  # [B, P, 1]
        page_ad_masks = ad_indices < num_page_ads_expanded  # [B, P, A]

        # 创建页面级掩码
        page_masks = torch.arange(page_size, device=device).view(1, -1) < num_pages.view(-1, 1)
        page_masks = page_masks.bool()

        # 收集所有点击物品
        all_clicks_rep = []  # 点击物品的新表示
        all_clicks_wi = []  # 信息熵权重
        all_clicks_ts = []  # 时间戳

        # 处理每个页面
        for b in range(batch_size):
            for p in range(page_size):
                if not page_masks[b, p]:
                    continue

                # 获取当前页面的广告和时间戳
                ads = page_seq[b, p]  # [A, D]
                ts = page_timestamps[b, p]  # [A]
                valid_mask = page_ad_masks[b, p]  # [A]
                click_mask = mask_click_ad[b, p]  # [A]

                # 分离点击和未点击物品
                clicked_indices = click_mask.nonzero(as_tuple=True)[0]
                unclicked_indices = (~click_mask & valid_mask).nonzero(as_tuple=True)[0]

                clicked_ads = ads[clicked_indices] if len(clicked_indices) > 0 else None
                clicked_ts = ts[clicked_indices] if len(clicked_indices) > 0 else None
                unclicked_ads = ads[unclicked_indices] if len(unclicked_indices) > 0 else None

                if clicked_ads is None or len(clicked_ads) == 0:
                    continue

                # 处理点击和未点击物品
                new_reps, wi_weights, timestamps = self._process_click_unclick(
                    clicked_ads,
                    unclicked_ads,
                    clicked_ts
                )

                all_clicks_rep.extend(new_reps)
                all_clicks_wi.extend(wi_weights)
                all_clicks_ts.extend(timestamps)

        # 如果没有点击物品，使用目标广告作为默认
        if len(all_clicks_rep) == 0:
            page_layer['user_interest'] = target
            return page_layer

        # 转换为张量
        all_clicks_rep = torch.stack(all_clicks_rep)  # [total_clicks, D]
        all_clicks_wi = torch.stack(all_clicks_wi)  # [total_clicks]
        all_clicks_ts = torch.stack(all_clicks_ts)  # [total_clicks]

        # 跨页面归一化信息熵权重
        global_wi_sum = all_clicks_wi.sum()
        normalized_wi = all_clicks_wi / (global_wi_sum + 1e-9)  # 归一化

        # 按时间戳排序点击物品
        sorted_indices = torch.argsort(all_clicks_ts)
        sorted_clicks = all_clicks_rep[sorted_indices]
        sorted_wi = normalized_wi[sorted_indices]
        sorted_ts = all_clicks_ts[sorted_indices]

        # 计算时间权重
        last_ts = sorted_ts[-1]
        delta_ts = last_ts - sorted_ts

        # 计算与最后点击物品的相似度
        last_click = sorted_clicks[-1]
        # 使用余弦相似度
        normalized_clicks = F.normalize(sorted_clicks, p=2, dim=1)
        normalized_last = F.normalize(last_click, p=2, dim=0)
        cos_sim = torch.mv(normalized_clicks, normalized_last)

        # 调整衰减参数
        lambda_adjusted = self.lambda_param * (1 - cos_sim)
        time_weights = torch.exp(-lambda_adjusted * delta_ts)
        time_weights = time_weights / (time_weights.sum() + 1e-9)  # 归一化

        # 动态融合权重
        wi_tw = torch.stack([sorted_wi, time_weights], dim=1)  # [total_clicks, 2]
        g = torch.sigmoid(self.gate_linear(wi_tw))  # [total_clicks, 1]
        gamma = g * sorted_wi.unsqueeze(1) + (1 - g) * time_weights.unsqueeze(1)
        gamma = gamma.squeeze()

        # 聚合用户兴趣
        user_interest = torch.sum(gamma.unsqueeze(1) * sorted_clicks, dim=0)  # [D]

        # 修复：确保用户兴趣向量与批次维度匹配
        # 创建一个与批次大小匹配的用户兴趣矩阵
        batch_size = embedding_layer['batch_size']
        feature_dim = user_interest.shape[0]
        user_interest_matrix = user_interest.unsqueeze(0).repeat(batch_size, 1)  # [B, D]

        page_layer['user_interest'] = user_interest_matrix
        return page_layer

    def _dnn_layer(self, embedding_layer, page_layer):
        dnn_layer = OrderedDict()
        # 使用用户兴趣向量替代原来的页面表示
        mlp_input = torch.cat([
            embedding_layer['user'],
            embedding_layer['target'],
            page_layer['user_interest']  # 现在这是 [B, D] 形状
        ], 1)  # [B, user_dim + target_dim + interest_dim]
        dnn_layer['dnn_out'] = self.dnn_net['dnn'](mlp_input)
        return dnn_layer

    def _logits_layer(self, dnn_layer):
        return self.logits_linear(dnn_layer['dnn_out'])

    def forward(self, features, epoch_id=0):
        embedding_layer = self._make_embedding_layer(features)
        page_layer = self._page_layer(embedding_layer)
        dnn_layer = self._dnn_layer(embedding_layer, page_layer)
        logits = self._logits_layer(dnn_layer)
        return logits.squeeze()

    def loss(self, logits, labels):
        loss = F.binary_cross_entropy_with_logits(logits.squeeze(), labels.float())
        return [loss]

    def init_weights(self):
        # 初始化新添加的层
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.xavier_uniform_(self.gate_linear.weight)
        nn.init.constant_(self.gate_linear.bias, 0)
        nn.init.constant_(self.lambda_param, 0.1)

        # 初始化融合MLP
        for layer in self.fusion_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        # 初始化原始嵌入层
        for key_name in ['user', 'ad', 'location', 'category']:
            for e in self.embedding_dict[key_name]:
                nn.init.xavier_uniform_(e.weight)
        for key_name in ['ad_title', 'ad_params', 'search_query', 'search_params', 'page_click_num']:

            nn.init.xavier_uniform_(self.embedding_dict[key_name].weight)
