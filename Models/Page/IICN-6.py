import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict

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

        # 初始化模块
        self._build_embedding_layer(self.num_features_dict)
        self.ad_embed_dim = self.cnt_fts_dict['ad_embed'] * self.embed_dim
        self.tad_embed_dim = (self.cnt_fts_dict['ad_embed'] - 2) * self.embed_dim
        self.qy_embed_dim = self.cnt_fts_dict['qy_embed'] * self.embed_dim
        self.adq_embed_dim = self.ad_embed_dim + self.qy_embed_dim

        # 新增论文功能模块
        self.willingness_weight = WillingnessWeight(self.embed_dim)
        self.unclicked_fusion = UnclickedFusion(self.embed_dim)
        self.time_weight = TimeWeight(self.embed_dim)
        self.weight_fusion_gate = WeightFusionGate(self.embed_dim)
        self.agsom = AGSOM(embed_dim=self.embed_dim)

        # DNN层
        dnn_input_dim = self.cnt_fts_dict['user'] * self.embed_dim + self.tad_embed_dim + self.embed_dim
        self.dnn_net = nn.ModuleDict({
            'dnn': MultiLayerPerceptron(dnn_input_dim, dnn_dim_list, dropout=0, activation=nn.PReLU(),
                                        output_layer=False)
        })
        self.logits_linear = nn.Linear(dnn_dim_list[-1], 1)
        self.init_weights()

    def _build_embedding_layer(self, num_features_dict):

        self.embedding_dict = nn.ModuleDict()
        self.cnt_fts_dict = OrderedDict()

        ### embedding uni value fts
        for key_name in ['user', 'ad', 'location', 'category']:
            num_features_list = num_features_dict[key_name]
            if key_name == 'ad':
                num_features_list = num_features_list[1:]  # delete search_id

            self.embedding_dict[key_name] = nn.ModuleList(nn.Embedding(x, self.embed_dim) for x in num_features_list)
            self.cnt_fts_dict[key_name] = len(num_features_list)

        ### embedding multi value fts
        for key_name in ['ad_title', 'ad_params', 'search_query', 'search_params'] + ['page_click_num']:
            self.embedding_dict[key_name] = nn.Embedding(num_features_dict[key_name], self.embed_dim)

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
        features_embed = torch.cat(features_embed_list, 1)
        return features_embed

    def __ad_embedding(self, ad_features, is_target=False):

        ad_embedding_dict = OrderedDict()
        loc_begin = 3
        ad_begin = 15
        fts_end = 33
        if is_target:
            fts_end -= 1
        begin_i = loc_begin

        for key_name in ['location', 'category']:
            tmp_features = ad_features[:, begin_i: begin_i + self.cnt_fts_dict[key_name]]
            ad_embedding_dict[key_name] = self.__feature_embedding(tmp_features, key_name)
            begin_i += self.cnt_fts_dict[key_name]

        for key_name in ['search_query', 'search_params']:
            tmp_features = ad_features[:, begin_i: begin_i + self.num_features_dict['multi'][key_name]]
            ad_embedding_dict[key_name] = nn.functional.embedding(tmp_features, self.embedding_dict[key_name].weight)
            begin_i += self.num_features_dict['multi'][key_name]

        begin_i = ad_begin + 1  # ad_id
        key_name = 'category'
        tmp_features = ad_features[:, begin_i: begin_i + self.cnt_fts_dict[key_name]]
        ad_embedding_dict['ad_' + key_name] = self.__feature_embedding(tmp_features, key_name)
        begin_i += self.cnt_fts_dict[key_name]

        for key_name in ['ad_title', 'ad_params']:
            tmp_features = ad_features[:, begin_i: begin_i + self.num_features_dict['multi'][key_name]]
            ad_embedding_dict[key_name] = nn.functional.embedding(tmp_features, self.embedding_dict[key_name].weight)
            begin_i += self.num_features_dict['multi'][key_name]

        tmp_features = torch.cat(
            (ad_features[:, :loc_begin], ad_features[:, ad_begin].view(-1, 1), ad_features[:, begin_i:fts_end]), 1)
        # ip, logged_on, timestamp, ad_id   , position, hist_ctr, is_click
        ad_embedding_dict['uni'] = self.__feature_embedding(tmp_features, 'ad')

        ad_features = torch.cat((
            ad_embedding_dict['uni'],  # [B, 5*D]
            ad_embedding_dict['location'],  # [B, 4*D]
            ad_embedding_dict['category'],  # [B, 4*D]
            ad_embedding_dict['search_query'].view(-1, self.embed_dim),  # [B, D]
            ad_embedding_dict['search_params'].sum(1).view(-1, self.embed_dim),  # [B, D]
            ad_embedding_dict['ad_category'],  # [B, 4*D]
            ad_embedding_dict['ad_title'].sum(1).view(-1, self.embed_dim),  # [B, D]
            ad_embedding_dict['ad_params'].sum(1).view(-1, self.embed_dim)  # [B, D]
        ), 1)
        timestamp = ad_features[:, 0]
        return ad_features, timestamp  # [B, 21*D]

    def __query_embedding(self, query_features):
        query_embedding_dict = OrderedDict()
        loc_begin = 3
        fts_end = 15
        begin_i = loc_begin

        for key_name in ['location', 'category']:
            tmp_features = query_features[:, begin_i: begin_i + self.cnt_fts_dict[key_name]]
            query_embedding_dict[key_name] = self.__feature_embedding(tmp_features, key_name)
            begin_i += self.cnt_fts_dict[key_name]

        for key_name in ['search_query', 'search_params']:
            tmp_features = query_features[:, begin_i: begin_i + self.num_features_dict['multi'][key_name]]
            query_embedding_dict[key_name] = nn.functional.embedding(tmp_features, self.embedding_dict[key_name].weight)
            begin_i += self.num_features_dict['multi'][key_name]

        tmp_features = query_features[:, :loc_begin]
        # ip, logged_on, timestamp,
        query_embedding_dict['uni'] = self.__feature_embedding(tmp_features, 'ad')

        query_features = torch.cat((
            query_embedding_dict['uni'],  # [B, 5*D]
            query_embedding_dict['location'],  # [B, 4*D]
            query_embedding_dict['category'],  # [B, 4*D]
            query_embedding_dict['search_query'].view(-1, self.embed_dim),  # [B, D]
            query_embedding_dict['search_params'].sum(1).view(-1, self.embed_dim),  # [B, D]
        ), 1)
        return query_features

    def _make_embedding_layer(self, features):
        ### 特征嵌入层
        embedding_layer = OrderedDict()
        batch_size = features.shape[0]
        embedding_layer['batch_size'] = batch_size

        cnt_user_fts = self.cnt_fts_dict['user']  # 5
        cnt_qad_fts = self.cnt_fts_dict['ad'] + self.cnt_fts_dict['location'] + \
                      self.cnt_fts_dict['category'] * 2 + self.cnt_fts_dict['multi']  # 7 + 12 + 14 = 34
        cnt_qpage_fts = cnt_qad_fts * 5 + 1 + 1  # + page_ad_num(1) + page_click_num(1)
        fts_index_bias = cnt_user_fts + cnt_qad_fts - 1  # -1: target w/o is_click

        embedding_layer['user'] = self.__feature_embedding(features[:, :cnt_user_fts], 'user')  # [B, 5*D]
        target_embed, _ = self.__ad_embedding(features[:, cnt_user_fts:fts_index_bias], is_target=True)
        embedding_layer['target'] = target_embed  # [B, *D]

        page_size = 5
        ad_size = 5
        query_size = 15
        page_seq = []
        query_seq = []
        num_page_ads = []
        timestamps = []  # 存储所有时间戳
        mask_nan_ad = torch.ones((batch_size, page_size, ad_size)).to(features.device)
        mask_click_ad = torch.ones((batch_size, page_size, ad_size)).to(features.device)

        for i in range(page_size):
            page_index_bias = fts_index_bias + i * cnt_qpage_fts
            ad_embed_list = []
            page_timestamps = []  # 当前页面的时间戳

            for j in range(ad_size):
                ad_slice = features[:, page_index_bias + j * cnt_qad_fts: page_index_bias + (j + 1) * cnt_qad_fts]
                ad_embed, timestamp = self.__ad_embedding(ad_slice)
                ad_embed_list.append(ad_embed)
                page_timestamps.append(timestamp)

                mask_nan_ad[:, i, j] = (ad_slice[:, -1] == 2)  # 假设最后一列是点击状态
                mask_click_ad[:, i, j] = (ad_slice[:, -1] == 1)  # 1表示点击

            # 存储当前页面的时间戳
            timestamps.append(torch.stack(page_timestamps, dim=1))  # [B, ad_size]

            # 构建当前页面的广告嵌入序列
            ad_embed_seq = torch.stack(ad_embed_list, 1)  # [B, ad_size, embed_dim]

            # 添加页面点击数量信息
            num_click = features[:, page_index_bias + ad_size * cnt_qad_fts + 1].view(-1)  # [B, ]
            num_click_a = self.embedding_dict['page_click_num'](num_click)  # [B, D]
            num_click_a = num_click_a.unsqueeze(1).repeat(1, ad_size, 1)  # [B, A, D]
            ad_embed_seq = torch.cat((ad_embed_seq, num_click_a), dim=2)
            page_seq.append(ad_embed_seq)

            # 查询嵌入
            query_embed = self.__query_embedding(features[:, page_index_bias: page_index_bias + query_size])  # [B, D]
            query_embed = torch.cat((query_embed, num_click_a[:, 0, :]), dim=1)  # 使用第一个广告的点击数量
            query_seq.append(query_embed)

            num_ad = features[:, page_index_bias + ad_size * cnt_qad_fts].view(-1)
            num_page_ads.append(num_ad)

        # 组织时间戳 [B, P, A]
        embedding_layer['timestamps'] = torch.stack(timestamps, dim=1)  # [B, page_size, ad_size]
        embedding_layer['page_seq'] = torch.stack(page_seq, 1)  # [B, page_size, ad_size, D]
        embedding_layer['query_seq'] = torch.stack(query_seq, 1)  # [B, page_size, D]
        embedding_layer['num_page_ads'] = torch.stack(num_page_ads, 1).view(batch_size, page_size)  # [B, page_size, ]
        embedding_layer['num_pages'] = features[:, -1].view(-1)
        embedding_layer['mask_nan_ad'] = mask_nan_ad.bool()
        embedding_layer['mask_click_ad'] = mask_click_ad.bool()
        return embedding_layer

    def _process_page(self, clicked_embeds, unclicked_embeds, timestamps):
        """处理单个页面，实现论文核心功能"""
        batch_size, max_clicks, embed_dim = clicked_embeds.shape

        # 创建有效掩码（时间戳不是无穷大表示有效）
        valid_mask = (timestamps != float('inf'))  # [B, max_clicks]

        # 1. 分配未点击物品
        assignment_dict = {}
        willingness_weights = torch.zeros(batch_size, max_clicks, device=clicked_embeds.device)
        enhanced_clicks = torch.zeros_like(clicked_embeds)

        for j in range(max_clicks):
            # 检查当前点击是否有效
            valid_click = valid_mask[:, j]

            for b in range(batch_size):
                if valid_click[b]:
                    d_j = clicked_embeds[b, j]

                    # 获取当前样本的未点击物品
                    sample_unclicked = unclicked_embeds[b]
                    sample_unclicked_valid = sample_unclicked[valid_mask[b]]  # 只考虑有效未点击

                    if len(sample_unclicked_valid) > 0:
                        # 计算相似度
                        sim_scores = F.cosine_similarity(
                            d_j.unsqueeze(0),
                            sample_unclicked_valid,
                            dim=-1
                        )

                        # 找到最相似的点击物品
                        max_sim, assign_idx = torch.max(sim_scores, dim=0)

                        # 记录分配结果
                        if j not in assignment_dict:
                            assignment_dict[j] = []
                        assignment_dict[j].append(assign_idx.item())
                    else:
                        assignment_dict[j] = []

                    # 计算意愿权重
                    # ... (这里需要实现意愿权重计算，但简化处理) ...
                    willingness_weights[b, j] = 1.0  # 简化处理，实际需要计算

                    # 融合未点击物品
                    # ... (简化处理) ...
                    enhanced_clicks[b, j] = d_j

        return enhanced_clicks, willingness_weights, timestamps


    def _page_layer(self, embedding_layer):
        page_seq = embedding_layer['page_seq']  # [B, P, A, D]
        mask_click_ad = embedding_layer['mask_click_ad']  # [B, P, A]
        timestamps = embedding_layer['timestamps']  # [B, P, A]
        batch_size, num_pages, num_ads, embed_dim = page_seq.shape

        # 收集所有页面的增强点击表示和权重
        all_enhanced_clicks = []
        all_willingness_weights = []
        all_timestamps = []

        # 首先找到批次中最大点击数量
        max_clicks = 0
        for p in range(num_pages):
            page_mask = mask_click_ad[:, p, :]  # [B, A]
            num_clicks_per_sample = page_mask.sum(dim=1)  # [B]
            max_clicks = max(max_clicks, num_clicks_per_sample.max().item())

        for p in range(num_pages):
            page_data = page_seq[:, p, :, :]  # [B, A, D]
            page_mask = mask_click_ad[:, p, :]  # [B, A]
            page_timestamps = timestamps[:, p, :]  # [B, A]

            # 分离点击和未点击 - 使用掩码索引
            clicked_embeds = []
            unclicked_embeds = []
            click_ts = []

            for i in range(batch_size):
                sample_click_mask = page_mask[i].bool()  # [A]
                sample_unclick_mask = ~sample_click_mask

                # 获取点击的嵌入和时间戳
                sample_clicked = page_data[i, sample_click_mask]  # [num_clicked, D]
                sample_unclicked = page_data[i, sample_unclick_mask]  # [num_unclicked, D]
                sample_ts = page_timestamps[i, sample_click_mask]  # [num_clicked]

                # 填充到最大长度
                padded_clicked = F.pad(sample_clicked, (0, 0, 0, max_clicks - sample_clicked.size(0)))
                padded_unclicked = F.pad(sample_unclicked, (0, 0, 0, max_clicks - sample_unclicked.size(0)))
                padded_ts = F.pad(sample_ts, (0, max_clicks - sample_ts.size(0)), value=float('inf'))

                clicked_embeds.append(padded_clicked)
                unclicked_embeds.append(padded_unclicked)
                click_ts.append(padded_ts)

            clicked_embeds = torch.stack(clicked_embeds)  # [B, max_clicks, D]
            unclicked_embeds = torch.stack(unclicked_embeds)  # [B, max_clicks, D]
            click_ts = torch.stack(click_ts)  # [B, max_clicks]

            # 处理页面
            enhanced_clicks, willingness_weights, click_timestamps = self._process_page(
                clicked_embeds, unclicked_embeds, click_ts
            )

            # 收集结果
            all_enhanced_clicks.append(enhanced_clicks)
            all_willingness_weights.append(willingness_weights)
            all_timestamps.append(click_timestamps)

        # 拼接所有页面的点击
        all_enhanced_clicks = torch.cat(all_enhanced_clicks, dim=1)  # [B, total_clicks, D]
        all_willingness_weights = torch.cat(all_willingness_weights, dim=1)  # [B, total_clicks]
        all_timestamps = torch.cat(all_timestamps, dim=1)  # [B, total_clicks]

        # 创建有效掩码（用于区分真实点击和填充）
        valid_mask = (all_timestamps != float('inf'))  # [B, total_clicks]

        # 5. 按时间排序
        sorted_indices = torch.argsort(all_timestamps, dim=1)

        # 扩展索引以匹配嵌入维度
        expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, embed_dim)

        sorted_clicks = torch.gather(all_enhanced_clicks, 1, expanded_indices)
        sorted_willingness = torch.gather(all_willingness_weights, 1, sorted_indices)
        sorted_timestamps = torch.gather(all_timestamps, 1, sorted_indices)
        sorted_valid_mask = torch.gather(valid_mask, 1, sorted_indices)

        # 6. 计算时间权重
        time_weights = self.time_weight(sorted_clicks, sorted_timestamps, sorted_valid_mask)

        # 7. 融合权重
        fused_weights = self.weight_fusion_gate(sorted_willingness, time_weights)

        # 8. 加权点击表示
        weighted_clicks = sorted_clicks * fused_weights.unsqueeze(-1)  # [B, total_clicks, D]

        # 9. AGSOM聚类 - 仅使用有效点击
        user_interest = []
        for i in range(batch_size):
            # 提取有效点击
            valid_indices = sorted_valid_mask[i]
            if valid_indices.any():
                valid_clicks = weighted_clicks[i, valid_indices]
                s = self.agsom(valid_clicks.unsqueeze(0))  # [1, D]
                user_interest.append(s)
            else:
                # 如果没有有效点击，使用零向量
                user_interest.append(torch.zeros(1, embed_dim, device=weighted_clicks.device))

        user_interest = torch.cat(user_interest, dim=0)  # [B, D]

        return {'user_interest': user_interest}


    def _dnn_layer(self, embedding_layer, page_layer):
        dnn_layer = OrderedDict()
        mlp_iput = torch.cat([
            embedding_layer['user'],  # 用户特征
            embedding_layer['target'],  # 目标广告特征
            page_layer['user_interest'],  # 用户兴趣表示
        ], 1)
        dnn_layer['dnn_out'] = self.dnn_net['dnn'](mlp_iput)
        return dnn_layer

    def forward(self, features, epoch_id=0):
        embedding_layer = self._make_embedding_layer(features)
        page_layer = self._page_layer(embedding_layer)
        dnn_layer = self._dnn_layer(embedding_layer, page_layer)
        logits = self._logits_layer(dnn_layer)
        return logits.squeeze()

    def loss(self, logtis, labels):
        loss = self.Loss(logtis.squeeze(), labels.float())
        return [loss]

    def init_weights(self):
        for key_name in ['user', 'ad', 'location', 'category']:
            for e in self.embedding_dict[key_name]:
                nn.init.xavier_uniform_(e.weight)
        for key_name in ['ad_title', 'ad_params', 'search_query', 'search_params']:
            nn.init.xavier_uniform_(self.embedding_dict[key_name].weight)

    def assign_unclicked_items(clicked_embeds, unclicked_embeds):
        """
        将未点击物品分配给最相似的点击物品
        :param clicked_embeds: 点击物品嵌入 [B, num_clicked, D]
        :param unclicked_embeds: 未点击物品嵌入 [B, num_unclicked, D]
        :return: 分配结果字典 {click_index: [unclick_indices]}
        """
        # 计算余弦相似度
        sim_matrix = F.cosine_similarity(
            unclicked_embeds.unsqueeze(2),  # [B, num_unclicked, 1, D]
            clicked_embeds.unsqueeze(1),  # [B, 1, num_clicked, D]
            dim=-1
        )  # [B, num_unclicked, num_clicked]

        # 找到每个未点击物品最相似的点击物品
        max_sim, assignments = torch.max(sim_matrix, dim=2)  # [B, num_unclicked]

        # 构建分配字典
        assignment_dict = {}
        for i in range(clicked_embeds.size(1)):
            mask = (assignments == i)
            assignment_dict[i] = mask

        return assignment_dict

# 将前面定义的功能模块类放在这里
class WillingnessWeight(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)

    def forward(self, clicked_embeds, unclicked_embeds, assignment_dict):
        """
        计算意愿权重
        :param clicked_embeds: 点击物品嵌入 [B, num_clicked, D]
        :param unclicked_embeds: 未点击物品嵌入 [B, num_unclicked, D]
        :param assignment_dict: 分配字典
        :return: 意愿权重 [B, num_clicked]
        """
        batch_size, num_clicked, _ = clicked_embeds.shape
        w_j = torch.zeros(batch_size, num_clicked, device=clicked_embeds.device)

        for j in range(num_clicked):
            # 当前点击物品
            d_j = clicked_embeds[:, j, :]

            # 获取分配给当前点击物品的未点击物品
            mask = assignment_dict[j]  # [B, num_unclicked]
            assigned_unclicked = unclicked_embeds * mask.unsqueeze(-1)

            # 计算相关度
            alpha_j = (self.Wq(d_j) * self.Wk(d_j)).sum(dim=-1)  # [B]

            # 计算与未点击物品的相关度
            alpha_jk = (self.Wq(d_j).unsqueeze(1) * self.Wk(assigned_unclicked)).sum(dim=-1)  # [B, num_unclicked]

            # 拼接所有相关度
            all_alphas = torch.cat([alpha_j.unsqueeze(1), alpha_jk], dim=1)  # [B, 1 + num_assigned]

            # Softmax标准化
            betas = F.softmax(all_alphas, dim=1)
            beta_j = betas[:, 0]
            beta_jk = betas[:, 1:]

            # 计算信息熵权重
            entropy_term = - (beta_j * torch.log2(beta_j + 1e-10))
            for k in range(beta_jk.size(1)):
                entropy_term -= beta_jk[:, k] * torch.log2(beta_jk[:, k] + 1e-10)

            w_j[:, j] = entropy_term

        # 跨页面归一化
        IE_j = w_j / w_j.sum(dim=1, keepdim=True)
        return IE_j


class UnclickedFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(4 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, d_j, unclicked_embeds):
        """
        融合点击物品与未点击物品
        :param d_j: 点击物品嵌入 [B, D]
        :param unclicked_embeds: 分配给该点击物品的未点击物品嵌入 [B, num_unclicked, D]
        :return: 融合后的表示 [B, D]
        """
        # 计算相关度
        alpha_jk = (self.Wq(d_j).unsqueeze(1) * self.Wk(unclicked_embeds)).sum(dim=-1)  # [B, num_unclicked]

        # 归一化权重
        mu_jk = F.softmax(alpha_jk, dim=1)  # [B, num_unclicked]

        # 计算上下文向量
        X_j = (mu_jk.unsqueeze(-1) * self.Wv(unclicked_embeds)).sum(dim=1)  # [B, D]

        # 融合点击物品和上下文
        concat_features = torch.cat([
            d_j,
            X_j,
            d_j - X_j,
            d_j * X_j
        ], dim=1)  # [B, 4*D]

        Y_j = self.mlp(concat_features)  # [B, D]
        return Y_j


class TimeWeight(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.lambda_param = nn.Parameter(torch.tensor(0.1))  # 可学习衰减参数

    def forward(self, click_embeds, timestamps, valid_mask):
        """
        计算时间权重
        :param click_embeds: 点击物品嵌入序列 [B, num_clicks, D]
        :param timestamps: 点击时间戳 [B, num_clicks]
        :param valid_mask: 有效点击掩码 [B, num_clicks]
        :return: 时间权重 [B, num_clicks]
        """
        batch_size, num_clicks, _ = click_embeds.shape

        # 计算时间差
        last_timestamp = timestamps[:, -1].unsqueeze(1)  # [B, 1]
        delta_t = last_timestamp - timestamps  # [B, num_clicks]

        # 计算与最后点击的相似度
        last_click = click_embeds[:, -1, :]  # [B, D]
        sim_scores = F.cosine_similarity(
            click_embeds,
            last_click.unsqueeze(1),
            dim=-1
        )  # [B, num_clicks]

        # 调整衰减参数
        lambda_adjusted = self.lambda_param * (1 - sim_scores)  # [B, num_clicks]

        # 计算基础权重
        base_weights = torch.exp(-lambda_adjusted * delta_t)  # [B, num_clicks]

        # 应用有效掩码，将无效位置的权重设为极小值
        base_weights = base_weights.masked_fill(~valid_mask, -1e9)

        # Softmax归一化
        time_weights = F.softmax(base_weights, dim=1)
        return time_weights

class WeightFusionGate(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.Wg = nn.Linear(2, 1)

    def forward(self, willingness_weights, time_weights):
        """
        融合意愿权重和时间权重
        :param willingness_weights: 意愿权重 [B, num_clicks]
        :param time_weights: 时间权重 [B, num_clicks]
        :return: 融合权重 [B, num_clicks]
        """
        # 拼接权重
        weights_concat = torch.stack([willingness_weights, time_weights], dim=-1)  # [B, num_clicks, 2]

        # 计算门控值
        g = torch.sigmoid(self.Wg(weights_concat)).squeeze(-1)  # [B, num_clicks]

        # 融合权重
        fused_weights = g * willingness_weights + (1 - g) * time_weights
        return fused_weights


class AGSOM(nn.Module):
    def __init__(self, grid_size=2, embed_dim=64, lr=0.01, sf=0.8, max_error=1.0):
        super().__init__()
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        self.lr = lr
        self.sf = sf
        self.max_error = max_error
        self.nodes = nn.Parameter(torch.randn(grid_size, grid_size, embed_dim))

    def forward(self, embeddings):
        """
        AGSOM聚类
        :param embeddings: 输入嵌入 [B, num_items, D]
        :return: 聚类中心之和 [B, D]
        """
        batch_size, num_items, _ = embeddings.shape
        results = torch.zeros(batch_size, self.embed_dim, device=embeddings.device)

        for b in range(batch_size):
            # 初始化
            grid = self.nodes.clone()
            cumulative_error = 0.0
            fd = 0.5  # 衰减因子

            for _ in range(3):  # 迭代3次
                max_err = 0.0
                max_err_item = None

                for i in range(num_items):
                    item_embed = embeddings[b, i]

                    # 1. 寻找最佳匹配节点
                    dists = torch.norm(grid.view(-1, self.embed_dim) - item_embed, dim=1)
                    min_dist, bmp_idx = torch.min(dists, dim=0)
                    bmp_coords = (bmp_idx // self.grid_size, bmp_idx % self.grid_size)

                    # 2. 更新节点
                    neighbors = self.get_neighbors(bmp_coords)
                    for coord in neighbors:
                        grid[coord] += self.lr * (item_embed - grid[coord])

                    # 3. 计算误差
                    err = min_dist.item()
                    if err > max_err:
                        max_err = err
                        max_err_item = item_embed

                    cumulative_error = fd * cumulative_error + err

                # 4. 判断是否增加节点
                gt = -self.embed_dim * math.log(self.sf)
                if max_err > gt:
                    # 简化的节点增加逻辑
                    if self.is_edge(bmp_coords):
                        # 实际实现中需要扩展网格
                        pass

                # 5. 判断是否删除节点
                if cumulative_error > self.max_error:
                    # 简化实现 - 实际需要根据命中次数删除
                    pass

            # 累加所有节点作为用户兴趣表示
            results[b] = grid.view(-1, self.embed_dim).sum(dim=0)

        return results

    def get_neighbors(self, coords):
        """获取邻居坐标"""
        x, y = coords
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                neighbors.append((nx, ny))
        return neighbors

    def is_edge(self, coords):
        """判断是否为边缘节点"""
        x, y = coords
        return (x == 0 or x == self.grid_size - 1 or
                y == 0 or y == self.grid_size - 1)

def assign_unclicked_items(clicked_embeds, unclicked_embeds):
        """
        将未点击物品分配给最相似的点击物品
        :param clicked_embeds: 点击物品嵌入 [B, num_clicked, D]
        :param unclicked_embeds: 未点击物品嵌入 [B, num_unclicked, D]
        :return: 分配结果字典 {click_index: [unclick_indices]}
        """
        # 计算余弦相似度
        sim_matrix = F.cosine_similarity(
            unclicked_embeds.unsqueeze(2),  # [B, num_unclicked, 1, D]
            clicked_embeds.unsqueeze(1),  # [B, 1, num_clicked, D]
            dim=-1
        )  # [B, num_unclicked, num_clicked]

        # 找到每个未点击物品最相似的点击物品
        max_sim, assignments = torch.max(sim_matrix, dim=2)  # [B, num_unclicked]

        # 构建分配字典
        assignment_dict = {}
        for i in range(clicked_embeds.size(1)):
            mask = (assignments == i)
            assignment_dict[i] = mask

        return assignment_dict

