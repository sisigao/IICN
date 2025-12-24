import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from Models.utils.layer import Attention, MultiLayerPerceptron


class RACP(nn.Module):
    def __init__(self, Sampler, ModelSettings):
        super().__init__()
        # 保存ModelSettings
        self.ModelSettings = ModelSettings

        # 初始化参数
        self.num_features_dict = Sampler.num_features_dict
        self.embed_dim = eval(ModelSettings['embed_dim'])
        dnn_dim_list = eval(ModelSettings['dnn_dim_list'])
        self.page_layer = ModelSettings['page_layer']
        self.remove_nan = eval(ModelSettings['remove_nan'])
        mha_head_num = eval(ModelSettings['mha_head_num'])

        # 构建嵌入层结构（不依赖具体数据）
        self._build_embedding_layer(self.num_features_dict)

        # 计算广告嵌入维度
        self.ad_embed_dim = self.cnt_fts_dict['ad_embed'] * self.embed_dim
        self.tad_embed_dim = (self.cnt_fts_dict['ad_embed'] - 2) * self.embed_dim

        # 物品嵌入维度
        self.item_embed_dim = 64
        self.agsom_k = 3

        # 投影层延迟初始化
        self.ad_proj = None
        self.target_embed_dim = None

        # 注意力参数
        self.W_Q = nn.Linear(self.item_embed_dim, self.item_embed_dim, bias=False)
        self.W_K = nn.Linear(self.item_embed_dim, self.item_embed_dim, bias=False)
        self.W_V = nn.Linear(self.item_embed_dim, self.item_embed_dim, bias=False)

        # 融合MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(4 * self.item_embed_dim, self.item_embed_dim),
            nn.ReLU()
        )

        # 时间参数
        self.lambda_param = nn.Parameter(torch.tensor(0.1))

        # 门控单元
        self.W_g = nn.Linear(2, 1, bias=True)

        # AGSOM参数
        self.agsom_nodes = nn.Parameter(
            torch.randn(self.agsom_k, self.item_embed_dim))
        self.agsom_lr = 0.1
        self.agsom_gt = -self.item_embed_dim * torch.log(torch.tensor(0.8))
        self.agsom_fd = 0.05
        self.agsom_max_error = nn.Parameter(torch.tensor(1.0))

        # 页面层
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
                'mha2': nn.MultiheadAttention(self.ad_embed_dim, num_heads=mha_head_num),
                'din2': Attention(self.ad_embed_dim, ModelSettings),
                'alpha2': MultiLayerPerceptron(alpha_input_dim, alpha_dim_list, dropout=0, activation=nn.ReLU(),
                                               output_layer=True),
            })
        else:
            raise ValueError('unknow PIN page layer name: ', self.page_layer)

        self.atten_net = torch.nn.MultiheadAttention(self.ad_embed_dim, num_heads=mha_head_num)

        # DNN输入维度将动态确定
        self.dnn_net = None
        self.logits_linear = None
        self.Loss = nn.BCEWithLogitsLoss()
        self.init_weights()

    def _build_embedding_layer(self, num_features_dict):
        self.embedding_dict = nn.ModuleDict()
        self.cnt_fts_dict = OrderedDict()

        for key_name in ['user', 'ad', 'location', 'category']:
            num_features_list = num_features_dict[key_name]
            if key_name == 'ad':
                num_features_list = num_features_list[1:]  # delete search_id
            self.embedding_dict[key_name] = nn.ModuleList(nn.Embedding(x, self.embed_dim) for x in num_features_list)
            self.cnt_fts_dict[key_name] = len(num_features_list)

        for key_name in ['ad_title', 'ad_params', 'search_query', 'search_params', 'page_click_num']:
            self.embedding_dict[key_name] = nn.Embedding(num_features_dict[key_name], self.embed_dim)

        # 特征数量统计
        self.cnt_fts_dict['multi'] = sum(num_features_dict['multi'].values())

        # 修复括号不匹配问题
        self.cnt_fts_dict['ad_embed'] = (
                self.cnt_fts_dict['ad'] +
                self.cnt_fts_dict['location'] +
                self.cnt_fts_dict['category'] * 2 +
                len(num_features_dict['multi']) + 1  # + page_click_num(1)
        )

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
            tmp_features = ad_features[:, begin_i: begin_i + self.cnt_fts_dict[key_name]]
            ad_embedding_dict[key_name] = self.__feature_embedding(tmp_features, key_name)
            begin_i += self.cnt_fts_dict[key_name]

        for key_name in ['search_query', 'search_params']:
            tmp_features = ad_features[:, begin_i: begin_i + self.num_features_dict['multi'][key_name]]
            ad_embedding_dict[key_name] = nn.functional.embedding(tmp_features, self.embedding_dict[key_name].weight)
            begin_i += self.num_features_dict['multi'][key_name]

        begin_i = ad_begin + 1
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
        return ad_features

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
        embedding_layer['target'] = self.__ad_embedding(features[:, cnt_user_fts:fts_index_bias], is_target=True)

        page_size = 5
        ad_size = 5
        query_size = 15
        page_seq = []
        query_seq = []
        num_page_ads = []
        mask_nan_ad = torch.ones((batch_size, page_size, ad_size)).to(features.device)
        mask_click_ad = torch.ones((batch_size, page_size, ad_size)).to(features.device)

        for i in range(page_size):
            page_index_bias = fts_index_bias + i * cnt_qpage_fts
            ad_embed_seq = [
                self.__ad_embedding(
                    features[:, page_index_bias + j * cnt_qad_fts: page_index_bias + (j + 1) * cnt_qad_fts])
                for j in range(ad_size)
            ]
            ad_embed_seq = torch.stack(ad_embed_seq, 1)
            num_click = features[:, page_index_bias + ad_size * cnt_qad_fts + 1].view(-1)
            num_click = self.embedding_dict['page_click_num'](num_click)
            num_click_a = num_click.unsqueeze(1).repeat(1, ad_size, 1)
            ad_embed_seq = torch.cat((ad_embed_seq, num_click_a), dim=2)
            page_seq.append(ad_embed_seq)

            query_embed = self.__query_embedding(features[:, page_index_bias: page_index_bias + query_size])
            query_embed = torch.cat((query_embed, num_click), dim=1)
            query_seq.append(query_embed)

            num_ad = features[:, page_index_bias + ad_size * cnt_qad_fts].view(-1)
            num_page_ads.append(num_ad)

            for j in range(ad_size):
                ad_features = features[:, page_index_bias + j * cnt_qad_fts: page_index_bias + (j + 1) * cnt_qad_fts]
                mask_nan_ad[:, i, j] = (ad_features[:, -1] == 2)
                mask_click_ad[:, i, j] = (ad_features[:, -1] == 1)

        embedding_layer['page_seq'] = torch.stack(page_seq, 1)
        embedding_layer['query_seq'] = torch.stack(query_seq, 1)
        embedding_layer['num_page_ads'] = torch.stack(num_page_ads, 1).view(batch_size, page_size)
        embedding_layer['num_pages'] = features[:, -1].view(-1)
        embedding_layer['mask_nan_ad'] = mask_nan_ad.bool()
        embedding_layer['mask_click_ad'] = mask_click_ad.bool()

        # 动态初始化投影层和DNN
        if self.ad_proj is None:
            # 获取目标广告的实际维度
            self.target_embed_dim = embedding_layer['target'].shape[1]
            device = embedding_layer['target'].device

            # 动态创建投影层
            self.ad_proj = nn.Linear(self.target_embed_dim, self.item_embed_dim).to(device)
            nn.init.xavier_uniform_(self.ad_proj.weight)
            if self.ad_proj.bias is not None:
                nn.init.constant_(self.ad_proj.bias, 0)

            # 动态创建DNN
            dnn_input_dim = (self.cnt_fts_dict['user'] * self.embed_dim +
                             self.target_embed_dim +
                             self.item_embed_dim)

            dnn_dim_list = eval(self.ModelSettings['dnn_dim_list'])
            self.dnn_net = nn.ModuleDict({
                'dnn': MultiLayerPerceptron(
                    dnn_input_dim, dnn_dim_list,
                    dropout=0, activation=nn.PReLU(),
                    output_layer=False
                )
            })
            self.logits_linear = nn.Linear(dnn_dim_list[-1], 1).to(device)

        return embedding_layer

    def _agsom_clustering(self, weighted_clicks):
        """
        AGSOM聚类算法
        weighted_clicks: [L, D] 加权后的点击物品表示
        """
        nodes = self.agsom_nodes.clone()
        L, D = weighted_clicks.shape

        # 1. 为每个物品找到最佳匹配节点
        for _ in range(3):  # 迭代3次
            for i in range(L):
                # 计算与所有节点的距离
                distances = torch.norm(weighted_clicks[i] - nodes, dim=1)
                bmp_idx = torch.argmin(distances)

                # 2. 更新节点
                neighbors = [bmp_idx]
                if bmp_idx > 0:
                    neighbors.append(bmp_idx - 1)
                if bmp_idx < nodes.shape[0] - 1:
                    neighbors.append(bmp_idx + 1)

                for idx in neighbors:
                    nodes[idx] += self.agsom_lr * (weighted_clicks[i] - nodes[idx])

        # 3. 计算误差并调整节点
        errors = []
        for i in range(L):
            distances = torch.norm(weighted_clicks[i] - nodes, dim=1)
            min_dist = torch.min(distances)
            errors.append(min_dist)

        max_error = torch.max(torch.stack(errors))
        if max_error > self.agsom_gt:
            # 添加新节点（简化实现）
            new_node = torch.mean(weighted_clicks, dim=0)
            nodes = torch.cat([nodes, new_node.unsqueeze(0)], dim=0)

        return nodes

    def _page_layer(self, embedding_layer):
        page_layer = OrderedDict()
        page_seq = embedding_layer['page_seq']
        device = page_seq.device
        batch_size = page_seq.shape[0]
        page_size = page_seq.shape[1]
        ad_size = page_seq.shape[2]
        num_page_ads = embedding_layer['num_page_ads']
        num_pages = embedding_layer['num_pages']
        mask_nan_ad = embedding_layer['mask_nan_ad']
        mask_click_ad = embedding_layer['mask_click_ad']

        # 创建广告和页面掩码
        ad_masks = [torch.arange(ad_size, device=device).view(1, -1) \
                        .repeat(page_size, 1) < num_page_ad.view(-1, 1) \
                    for num_page_ad in num_page_ads]
        page_ad_masks = torch.stack(ad_masks).bool()
        page_masks = torch.arange(page_size, device=device).view(1, -1) \
                         .repeat(batch_size, 1) < num_pages.view(-1, 1)
        page_masks = page_masks.bool()

        # 投影广告表示 - 只取主要特征部分
        if self.target_embed_dim is None:
            # 如果还未初始化，使用完整维度
            projected_page_seq = self.ad_proj(page_seq)
        else:
            # 只取主要特征部分
            raw_ads = page_seq[..., :self.target_embed_dim]
            projected_page_seq = self.ad_proj(raw_ads)

        # 处理每个页面
        all_s = []
        for b in range(batch_size):
            all_click_reps = []
            all_click_times = []
            all_click_weights = []

            for i in range(page_size):
                if not page_masks[b, i]:
                    continue

                ads = projected_page_seq[b, i]
                valid_mask = page_ad_masks[b, i]
                click_mask = mask_click_ad[b, i]

                ads = ads[valid_mask]
                click_mask = click_mask[valid_mask]

                clicked_ads = ads[click_mask]
                non_clicked_ads = ads[~click_mask]

                if clicked_ads.shape[0] == 0:
                    continue

                # 1. 计算相似度并分配未点击物品
                clicked_norm = F.normalize(clicked_ads, p=2, dim=1)
                non_clicked_norm = F.normalize(non_clicked_ads, p=2, dim=1)
                sim_matrix = torch.mm(non_clicked_norm, clicked_norm.t())
                _, max_indices = torch.max(sim_matrix, dim=1)

                # 2. 计算信息熵权重
                alpha_js = []
                alpha_jks_list = []

                for j in range(clicked_ads.shape[0]):
                    d_j = clicked_ads[j]
                    q_j = self.W_Q(d_j)
                    k_j = self.W_K(d_j)
                    alpha_j = torch.dot(q_j, k_j)
                    alpha_js.append(alpha_j)

                    mask = (max_indices == j)
                    assigned_ads = non_clicked_ads[mask]

                    alpha_jks = []
                    for u_k in assigned_ads:
                        k_k = self.W_K(u_k)
                        alpha_jk = torch.dot(q_j, k_k)
                        alpha_jks.append(alpha_jk)
                    alpha_jks_list.append(alpha_jks)

                # 3. 计算标准化权重和信息熵
                w_js = []
                new_clicked_ads = []

                for j in range(len(alpha_js)):
                    all_alphas = [alpha_js[j]] + alpha_jks_list[j]
                    all_alphas_tensor = torch.tensor(all_alphas, device=device)
                    beta = F.softmax(all_alphas_tensor, dim=0)

                    beta_j = beta[0]
                    beta_jks = beta[1:]

                    # 计算信息熵
                    entropy = 0.0
                    if beta_j > 1e-5:
                        entropy -= beta_j * torch.log2(beta_j)

                    for beta_jk in beta_jks:
                        if beta_jk > 1e-5:
                            entropy -= beta_jk * torch.log2(beta_jk)

                    w_js.append(entropy)

                    # 4. 融合未点击物品
                    if len(alpha_jks_list[j]) > 0:
                        alpha_jks_tensor = torch.tensor(alpha_jks_list[j], device=device)
                        mu_jks = F.softmax(alpha_jks_tensor, dim=0)

                        u_ks = non_clicked_ads[mask]
                        v_ks = self.W_V(u_ks)
                        weighted_sum = torch.sum(mu_jks.unsqueeze(1) * v_ks, dim=0)

                        # 拼接特征
                        concat_vec = torch.cat([
                            clicked_ads[j],
                            weighted_sum,
                            clicked_ads[j] - weighted_sum,
                            clicked_ads[j] * weighted_sum
                        ], dim=0)

                        new_d_j = self.fusion_mlp(concat_vec)
                    else:
                        new_d_j = clicked_ads[j]

                    new_clicked_ads.append(new_d_j)

                # 5. 存储结果
                time_stamps = [i * 10 + j for j in range(len(new_clicked_ads))]
                all_click_reps.extend(new_clicked_ads)
                all_click_times.extend(time_stamps)
                all_click_weights.extend(w_js)

            # 6. 处理时间权重和融合
            if len(all_click_reps) > 0:
                sorted_indices = sorted(range(len(all_click_times)),
                                        key=lambda k: all_click_times[k])
                sorted_click_reps = torch.stack([all_click_reps[i] for i in sorted_indices])
                sorted_click_weights = torch.tensor(
                    [all_click_weights[i] for i in sorted_indices], device=device)
                sorted_time_diffs = torch.tensor(
                    [all_click_times[sorted_indices[-1]] - all_click_times[i]
                     for i in sorted_indices], device=device, dtype=torch.float)

                # 计算时间权重
                base_weights = torch.exp(-self.lambda_param * sorted_time_diffs)
                last_click_rep = sorted_click_reps[-1]
                sims = F.cosine_similarity(
                    sorted_click_reps, last_click_rep.unsqueeze(0), dim=1)
                lambda_adjusted = self.lambda_param * (1 - sims)
                time_weights = torch.exp(-lambda_adjusted * sorted_time_diffs)
                time_weights_normalized = F.softmax(time_weights, dim=0)

                # 归一化意愿权重
                ie_weights_normalized = sorted_click_weights / sorted_click_weights.sum()

                # 门控融合
                weight_matrix = torch.stack([
                    ie_weights_normalized,
                    time_weights_normalized
                ], dim=1)
                g = torch.sigmoid(self.W_g(weight_matrix)).squeeze()
                gamma_s = g * ie_weights_normalized + (1 - g) * time_weights_normalized

                # 应用权重
                weighted_click_reps = gamma_s.unsqueeze(1) * sorted_click_reps

                # AGSOM聚类
                agsom_nodes = self._agsom_clustering(weighted_click_reps)
                s = torch.sum(agsom_nodes, dim=0)
            else:
                s = torch.zeros(self.item_embed_dim, device=device)

            all_s.append(s)

        # 返回用户兴趣表示
        page_layer['page_rep'] = torch.stack(all_s, dim=0)
        return page_layer

    def _dnn_layer(self, embedding_layer, page_layer):
        target_proj = self.ad_proj(embedding_layer['target'])

        mlp_input = torch.cat([
            embedding_layer['user'],
            target_proj,
            page_layer['page_rep']
        ], 1)

        dnn_layer = OrderedDict()
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

    def loss(self, logtis, labels):
        loss = self.Loss(logtis.squeeze(), labels.float())
        return [loss]

    def init_weights(self):
        for key_name in ['user', 'ad', 'location', 'category']:
            for e in self.embedding_dict[key_name]:
                nn.init.xavier_uniform_(e.weight)
        for key_name in ['ad_title', 'ad_params', 'search_query', 'search_params', 'page_click_num']:
            nn.init.xavier_uniform_(self.embedding_dict[key_name].weight)

        # 初始化新添加的层
        if hasattr(self, 'W_Q'):
            nn.init.xavier_uniform_(self.W_Q.weight)
        if hasattr(self, 'W_K'):
            nn.init.xavier_uniform_(self.W_K.weight)
        if hasattr(self, 'W_V'):
            nn.init.xavier_uniform_(self.W_V.weight)
        if hasattr(self, 'W_g'):
            nn.init.xavier_uniform_(self.W_g.weight)
            if self.W_g.bias is not None:
                nn.init.constant_(self.W_g.bias, 0)
        if hasattr(self, 'fusion_mlp'):
            for layer in self.fusion_mlp:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

        # 动态初始化的层会在第一次前向传播时初始化