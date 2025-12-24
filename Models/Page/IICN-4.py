import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
from Models.utils.layer import Attention, MultiLayerPerceptron


class IICN(nn.Module):
    def __init__(self, Sampler, ModelSettings):
        super().__init__()

        # Initialize arguments
        self.num_features_dict = Sampler.num_features_dict
        self.embed_dim = eval(ModelSettings['embed_dim'])
        dnn_dim_list = eval(ModelSettings['dnn_dim_list'])
        self.page_layer = ModelSettings['page_layer']
        self.remove_nan = eval(ModelSettings['remove_nan'])
        mha_head_num = eval(ModelSettings['mha_head_num'])

        # Initialize model layers
        self._build_embedding_layer(self.num_features_dict)
        self.ad_embed_dim = self.cnt_fts_dict['ad_embed'] * self.embed_dim
        self.tad_embed_dim = (self.cnt_fts_dict['ad_embed'] - 2) * self.embed_dim
        self.qy_embed_dim = self.cnt_fts_dict['qy_embed'] * self.embed_dim
        self.adq_embed_dim = self.ad_embed_dim + self.qy_embed_dim

        # 1. 添加新组件用于未点击物品处理和时间衰减
        self.click_enhance_mlp = nn.Sequential(
            nn.Linear(4 * self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        self.fusion_gate = nn.Linear(2, 1)  # 用于融合意愿和时间权重
        self.lambda_param = nn.Parameter(torch.tensor(0.1))  # 可学习的时间衰减参数

        if self.page_layer == 'dynamic_page':
            alpha_input_dim = self.ad_embed_dim + self.tad_embed_dim
            alpha_dim_list = eval(ModelSettings['alpha_dim_list'])
            self.page_net = nn.ModuleDict({
                'target_to_adq': nn.Sequential(
                    nn.Linear(self.tad_embed_dim, self.ad_embed_dim), nn.ReLU()
                ),
                'target_to_pq': nn.Sequential(
                    nn.Linear(self.tad_embed_dim, self.ad_embed_dim), nn.ReLU()
                ),
            })
        else:
            raise ValueError('unknown PIN page layer name: ', self.page_layer)

        self.atten_net = nn.MultiheadAttention(self.ad_embed_dim, num_heads=mha_head_num)

        dnn_input_dim = self.cnt_fts_dict['user'] * self.embed_dim + self.tad_embed_dim + self.embed_dim
        self.dnn_net = nn.ModuleDict({
            'dnn': MultiLayerPerceptron(dnn_input_dim, dnn_dim_list, dropout=0,
                                        activation=nn.PReLU(), output_layer=False)
        })

        self.logits_linear = nn.Linear(dnn_dim_list[-1], 1)
        self.init_weights()

    def _build_embedding_layer(self, num_features_dict):
        self.embedding_dict = nn.ModuleDict()
        self.cnt_fts_dict = OrderedDict()

        # Embedding for single-value features
        for key_name in ['user', 'ad', 'location', 'category']:
            num_features_list = num_features_dict[key_name]
            if key_name == 'ad':
                num_features_list = num_features_list[1:]
            self.embedding_dict[key_name] = nn.ModuleList(
                [nn.Embedding(x, self.embed_dim) for x in num_features_list])
            self.cnt_fts_dict[key_name] = len(num_features_list)

        # Embedding for multi-value features
        for key_name in ['ad_title', 'ad_params', 'search_query', 'search_params'] + ['page_click_num']:
            self.embedding_dict[key_name] = nn.Embedding(num_features_dict[key_name], self.embed_dim)

        # Count feature types
        self.cnt_fts_dict['multi'] = sum(num_features_dict['multi'].values())
        self.cnt_fts_dict['ad_embed'] = (self.cnt_fts_dict['ad'] + self.cnt_fts_dict['location'] +
                                         self.cnt_fts_dict['category'] * 2 +
                                         len(num_features_dict['multi'].values()) + 1)
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
            fts_end -= 1
        begin_i = loc_begin

        for key_name in ['location', 'category']:
            tmp_features = ad_features[:, begin_i:begin_i + self.cnt_fts_dict[key_name]]
            ad_embedding_dict[key_name] = self.__feature_embedding(tmp_features, key_name)
            begin_i += self.cnt_fts_dict[key_name]

        for key_name in ['search_query', 'search_params']:
            tmp_features = ad_features[:, begin_i:begin_i + self.num_features_dict['multi'][key_name]]
            ad_embedding_dict[key_name] = F.embedding(tmp_features, self.embedding_dict[key_name].weight)
            begin_i += self.num_features_dict['multi'][key_name]

        begin_i = ad_begin + 1  # ad_id
        key_name = 'category'
        tmp_features = ad_features[:, begin_i:begin_i + self.cnt_fts_dict[key_name]]
        ad_embedding_dict['ad_' + key_name] = self.__feature_embedding(tmp_features, key_name)
        begin_i += self.cnt_fts_dict[key_name]

        for key_name in ['ad_title', 'ad_params']:
            tmp_features = ad_features[:, begin_i:begin_i + self.num_features_dict['multi'][key_name]]
            ad_embedding_dict[key_name] = F.embedding(tmp_features, self.embedding_dict[key_name].weight)
            begin_i += self.num_features_dict['multi'][key_name]

        # 在目标广告中，is_click字段不存在
        if is_target:
            is_click = torch.zeros_like(ad_features[:, 0])
        else:
            is_click = ad_features[:, -1]

        tmp_features = torch.cat((ad_features[:, :loc_begin],
                                  ad_features[:, ad_begin].view(-1, 1),
                                  ad_features[:, begin_i:fts_end]), 1)
        ad_embedding_dict['uni'] = self.__feature_embedding(tmp_features, 'ad')

        ad_features_embed = torch.cat((
            ad_embedding_dict['uni'],
            ad_embedding_dict['location'],
            ad_embedding_dict['category'],
            ad_embedding_dict['search_query'].view(-1, self.embed_dim),
            ad_embedding_dict['search_params'].sum(1).view(-1, self.embed_dim),
            ad_embedding_dict['ad_category'],
            ad_embedding_dict['ad_title'].sum(1).view(-1, self.embed_dim),
            ad_embedding_dict['ad_params'].sum(1).view(-1, self.embed_dim)
        ), 1)

        return ad_features_embed, is_click

    def __query_embedding(self, query_features):
        # 2. 正确地从查询特征中提取时间戳（索引3）
        timestamp = query_features[:, 3]  # 时间戳是查询特征的第4个字段（索引为3）

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
            query_embedding_dict[key_name] = F.embedding(tmp_features, self.embedding_dict[key_name].weight)
            begin_i += self.num_features_dict['multi'][key_name]

        tmp_features = query_features[:, :loc_begin]
        query_embedding_dict['uni'] = self.__feature_embedding(tmp_features, 'ad')

        query_embed = torch.cat((
            query_embedding_dict['uni'],
            query_embedding_dict['location'],
            query_embedding_dict['category'],
            query_embedding_dict['search_query'].view(-1, self.embed_dim),
            query_embedding_dict['search_params'].sum(1).view(-1, self.embed_dim),
        ), 1)

        return query_embed, timestamp

    def _make_embedding_layer(self, features):
        embedding_layer = OrderedDict()
        batch_size = features.shape[0]
        embedding_layer['batch_size'] = batch_size

        cnt_user_fts = self.cnt_fts_dict['user']
        cnt_qad_fts = (self.cnt_fts_dict['ad'] + self.cnt_fts_dict['location'] +
                       self.cnt_fts_dict['category'] * 2 + self.cnt_fts_dict['multi'])
        cnt_qpage_fts = cnt_qad_fts * 5 + 1 + 1
        fts_index_bias = cnt_user_fts + cnt_qad_fts - 1

        embedding_layer['user'] = self.__feature_embedding(features[:, :cnt_user_fts], 'user')
        target_embed, _ = self.__ad_embedding(features[:, cnt_user_fts:fts_index_bias], is_target=True)
        embedding_layer['target'] = target_embed

        page_size = 5
        ad_size = 5
        query_size = 15
        page_seq = []
        query_seq = []
        num_page_ads = []
        page_timestamps = []  # 存储页面时间戳
        page_is_click = []

        mask_nan_ad = torch.ones((batch_size, page_size, ad_size), device=features.device)
        mask_click_ad = torch.ones((batch_size, page_size, ad_size), device=features.device)

        # 3. 收集时间戳和点击信息
        for i in range(page_size):
            page_index_bias = fts_index_bias + i * cnt_qpage_fts

            # 先处理查询特征，获取时间戳
            query_start = page_index_bias
            query_end = page_index_bias + query_size
            query_embed, page_timestamp = self.__query_embedding(features[:, query_start:query_end])

            # 存储页面时间戳
            page_timestamps.append(page_timestamp)

            page_is_click.append([])
            page_ads = []

            # 处理该页面的5个广告
            for j in range(ad_size):
                start_idx = query_end + j * cnt_qad_fts
                end_idx = query_end + (j + 1) * cnt_qad_fts
                ad_embed, is_click = self.__ad_embedding(features[:, start_idx:end_idx])
                page_ads.append(ad_embed)
                page_is_click[i].append(is_click)

                # 更新掩码
                ad_features = features[:, start_idx:end_idx]
                mask_nan_ad[:, i, j] = (ad_features[:, -1] == 2)
                mask_click_ad[:, i, j] = (ad_features[:, -1] == 1)

            page_ads = torch.stack(page_ads, 1)
            num_click = features[:, page_index_bias + ad_size * cnt_qad_fts + 1].view(-1)
            num_click_embed = self.embedding_dict['page_click_num'](num_click)
            num_click_a = num_click_embed.unsqueeze(1).repeat(1, ad_size, 1)
            page_ads = torch.cat((page_ads, num_click_a), dim=2)
            page_seq.append(page_ads)

            query_embed = torch.cat((query_embed, num_click_embed), dim=1)
            query_seq.append(query_embed)

            num_ad = features[:, page_index_bias + ad_size * cnt_qad_fts].view(-1)
            num_page_ads.append(num_ad)

        embedding_layer['page_seq'] = torch.stack(page_seq, 1)
        embedding_layer['query_seq'] = torch.stack(query_seq, 1)
        embedding_layer['num_page_ads'] = torch.stack(num_page_ads, 1).view(batch_size, page_size)
        embedding_layer['num_pages'] = features[:, -1].view(-1)
        embedding_layer['mask_nan_ad'] = mask_nan_ad.bool()
        embedding_layer['mask_click_ad'] = mask_click_ad.bool()

        # 4. 存储时间戳和点击信息
        embedding_layer['page_timestamps'] = torch.stack(page_timestamps, 1)  # [B, P]
        embedding_layer['page_is_click'] = torch.stack(
            [torch.stack(ic, dim=1) for ic in page_is_click], dim=1)  # [B, P, A]

        return embedding_layer

    def _process_page(self, page_embed, page_ts, page_ic, page_mask):
        """处理单个页面，实现论文中的核心逻辑"""
        batch_size, ad_size, embed_dim = page_embed.shape
        device = page_embed.device

        # 初始化结果容器
        enhanced_clicks = []
        ie_weights = []
        click_timestamps = []

        for b in range(batch_size):
            # 提取有效广告
            valid_mask = page_mask[b]
            if not valid_mask.any():
                continue

            valid_embed = page_embed[b][valid_mask]
            valid_ts = page_ts[b]  # 页面时间戳（所有广告共享）
            valid_ic = page_ic[b][valid_mask]

            # 分离点击和未点击
            clicked_mask = valid_ic == 1
            clicked_embed = valid_embed[clicked_mask]
            clicked_ts = torch.full_like(clicked_mask, valid_ts, dtype=torch.float)  # 所有点击共享页面时间戳
            unclicked_embed = valid_embed[~clicked_mask]

            if len(clicked_embed) == 0:
                continue

            # 5. 为每个未点击物品找到最相似的点击物品
            if len(unclicked_embed) > 0:
                clicked_norm = F.normalize(clicked_embed, p=2, dim=1)
                unclicked_norm = F.normalize(unclicked_embed, p=2, dim=1)
                sim_matrix = torch.mm(unclicked_norm, clicked_norm.t())
                _, cluster_assign = torch.max(sim_matrix, dim=1)
            else:
                cluster_assign = torch.tensor([], device=device)

            # 6. 处理每个点击物品及其分配的未点击物品
            for c_idx in range(len(clicked_embed)):
                # 获取当前点击物品及其分配的未点击物品
                c_embed = clicked_embed[c_idx]
                c_ts = clicked_ts[c_idx]  # 页面时间戳

                # 查找属于当前类的未点击物品
                if len(unclicked_embed) > 0:
                    class_mask = cluster_assign == c_idx
                    u_embeds = unclicked_embed[class_mask]
                else:
                    u_embeds = None

                # 计算相关度
                alpha_j = torch.dot(c_embed, c_embed).unsqueeze(0)
                if u_embeds is not None and len(u_embeds) > 0:
                    alpha_jk = torch.mv(u_embeds, c_embed)
                else:
                    alpha_jk = torch.tensor([], device=device)

                # 7. 归一化相关度
                if len(alpha_jk) > 0:
                    all_alphas = torch.cat([alpha_j, alpha_jk])
                    beta_all = F.softmax(all_alphas, dim=0)
                    beta_j = beta_all[0]
                    beta_jk = beta_all[1:]
                else:
                    beta_j = torch.tensor(1.0, device=device)
                    beta_jk = torch.tensor([], device=device)

                # 8. 计算信息熵权重
                entropy = 0.0
                if beta_j > 0:
                    entropy -= beta_j * torch.log2(beta_j + 1e-10)
                for beta in beta_jk:
                    if beta > 0:
                        entropy -= beta * torch.log2(beta + 1e-10)
                w_j = entropy

                # 9. 融合未点击物品
                if u_embeds is not None and len(u_embeds) > 0:
                    mu_jk = F.softmax(alpha_jk, dim=0)
                    X_j = torch.sum(mu_jk.view(-1, 1) * u_embeds, dim=0)
                else:
                    X_j = torch.zeros_like(c_embed)

                # 10. 构建增强表示
                concat_vec = torch.cat([
                    c_embed,
                    X_j,
                    c_embed - X_j,
                    c_embed * X_j
                ], dim=0)
                Y_j = self.click_enhance_mlp(concat_vec)

                # 存储结果
                enhanced_clicks.append(Y_j)
                ie_weights.append(w_j)
                click_timestamps.append(c_ts)

        return enhanced_clicks, ie_weights, click_timestamps

    def _page_layer(self, embedding_layer):
        page_layer = OrderedDict()
        page_seq = embedding_layer['page_seq']  # [B, P, A, D]
        page_ts = embedding_layer['page_timestamps']  # [B, P] 页面时间戳
        page_ic = embedding_layer['page_is_click']  # [B, P, A]
        num_page_ads = embedding_layer['num_page_ads']  # [B, P]
        num_pages = embedding_layer['num_pages']  # [B]
        page_mask = embedding_layer['mask_nan_ad']  # [B, P, A]

        batch_size, page_size, ad_size, embed_dim = page_seq.shape
        device = page_seq.device

        # 11. 收集所有页面的点击物品
        all_enhanced_clicks = []
        all_ie_weights = []
        all_click_ts = []

        for p in range(page_size):
            for b in range(batch_size):
                # 只处理有效页面
                if p >= num_pages[b]:
                    continue

                # 处理单个页面
                page_embed = page_seq[b, p]  # [A, D]
                page_ts_val = page_ts[b, p]  # 标量值
                page_ic_val = page_ic[b, p]  # [A]
                ad_mask = page_mask[b, p]  # [A]

                enhanced, ie_weights, click_ts = self._process_page(
                    page_embed, page_ts_val, page_ic_val, ad_mask)

                all_enhanced_clicks.extend(enhanced)
                all_ie_weights.extend(ie_weights)
                all_click_ts.extend(click_ts)

        # 12. 如果没有点击，返回零向量
        if len(all_enhanced_clicks) == 0:
            page_layer['page_rep'] = torch.zeros(batch_size, embed_dim, device=device)
            return page_layer

        # 转换为张量
        all_enhanced_clicks = torch.stack(all_enhanced_clicks)  # [T, D]
        all_ie_weights = torch.stack(all_ie_weights)  # [T]
        all_click_ts = torch.stack(all_click_ts)  # [T]

        # 13. 归一化信息熵权重
        ie_weights_norm = all_ie_weights / (all_ie_weights.sum() + 1e-10)

        # 14. 按时间排序
        sorted_indices = torch.argsort(all_click_ts)
        sorted_clicks = all_enhanced_clicks[sorted_indices]
        sorted_ie = ie_weights_norm[sorted_indices]
        sorted_ts = all_click_ts[sorted_indices]

        # 15. 计算时间权重
        t_last = sorted_ts[-1]
        delta_ts = t_last - sorted_ts

        # 计算相似度调整衰减因子
        last_click = sorted_clicks[-1].unsqueeze(0)
        sims = F.cosine_similarity(sorted_clicks, last_click, dim=1)
        lambda_adjusted = self.lambda_param * (1 - sims)

        # 计算时间权重
        time_weights = torch.exp(-lambda_adjusted * delta_ts)
        time_weights = time_weights / (time_weights.sum() + 1e-10)

        # 16. 动态融合权重
        gate_input = torch.stack([sorted_ie, time_weights], dim=1)
        g = torch.sigmoid(self.fusion_gate(gate_input))
        gamma = g * sorted_ie + (1 - g) * time_weights

        # 17. 加权聚合
        weighted_clicks = gamma.unsqueeze(1) * sorted_clicks
        user_interest = weighted_clicks.sum(dim=0)

        page_layer['page_rep'] = user_interest
        return page_layer

    def _dnn_layer(self, embedding_layer, page_layer):
        dnn_layer = OrderedDict()
        mlp_input = torch.cat([
            embedding_layer['user'],
            embedding_layer['target'],
            page_layer['page_rep'],
        ], 1)
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
        return [F.binary_cross_entropy_with_logits(logits.squeeze(), labels.float())]

    def init_weights(self):
        for key_name in ['user', 'ad', 'location', 'category']:
            for e in self.embedding_dict[key_name]:
                nn.init.xavier_uniform_(e.weight)
        for key_name in ['ad_title', 'ad_params', 'search_query', 'search_params']:
            nn.init.xavier_uniform_(self.embedding_dict[key_name].weight)
        # 初始化新添加的模块
        for m in [self.click_enhance_mlp, self.fusion_gate]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

                nn.init.constant_(m.bias, 0)
