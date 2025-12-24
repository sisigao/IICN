import torch
import torch.nn as nn
import torch.nn.functional as F 

from collections import OrderedDict
from Models.utils.layer import Attention, MultiLayerPerceptron


class IICN(nn.Module):
    def __init__(self, Sampler, ModelSettings):
        super().__init__()

        # init args
        self.num_features_dict = Sampler.num_features_dict
        self.embed_dim = eval(ModelSettings['embed_dim'])
        dnn_dim_list = eval(ModelSettings['dnn_dim_list'])
        self.page_layer = ModelSettings['page_layer']
        self.remove_nan = eval(ModelSettings['remove_nan'])
        mha_head_num = eval(ModelSettings['mha_head_num'])

        # init model layer
        self._build_embedding_layer(self.num_features_dict)  # build embeeding and cnt_*_fts
        self.ad_embed_dim = self.cnt_fts_dict['ad_embed'] * self.embed_dim
        self.tad_embed_dim = (self.cnt_fts_dict['ad_embed'] - 2) * self.embed_dim  # remove is_click and page_click_num
        self.qy_embed_dim = self.cnt_fts_dict['qy_embed'] * self.embed_dim
        self.adq_embed_dim = self.ad_embed_dim + self.qy_embed_dim

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
                    # nn.Linear(self.tad_embed_dim, self.adq_embed_dim), nn.ReLU()
                    nn.Linear(self.tad_embed_dim, self.ad_embed_dim), nn.ReLU()
                ),
                'mha2': nn.MultiheadAttention(self.adq_embed_dim, num_heads=mha_head_num),
                'din2': Attention(self.ad_embed_dim, ModelSettings),
                # 'din2': Attention(self.adq_embed_dim, ModelSettings),
                'alpha2': MultiLayerPerceptron(alpha_input_dim, alpha_dim_list, dropout=0, activation=nn.ReLU(),
                                               output_layer=True),

            })
        else:
            raise ValueError('unknow PIN page layer name: ', self.page_layer)

        self.atten_net = torch.nn.MultiheadAttention(self.ad_embed_dim, num_heads=mha_head_num)

        dnn_input_dim = self.cnt_fts_dict['user'] * self.embed_dim + self.tad_embed_dim + self.ad_embed_dim
        # dnn_input_dim = self.cnt_fts_dict['user']*self.embed_dim + self.tad_embed_dim + self.ad_embed_dim + self.qy_embed_dim
        self.dnn_net = nn.ModuleDict({
            # 'dnn_input_bn': nn.BatchNorm1d(dnn_input_dim),
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

        return ad_features  # [B, 21*D]

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
        ###  Embedding Layer
        embedding_layer = OrderedDict()
        batch_size = features.shape[0]
        embedding_layer['batch_size'] = batch_size

        cnt_user_fts = self.cnt_fts_dict['user']  # 5
        cnt_qad_fts = self.cnt_fts_dict['ad'] + self.cnt_fts_dict['location'] + \
                      self.cnt_fts_dict['category'] * 2 + self.cnt_fts_dict['multi']  # 7 + 12 + 14 = 34
        cnt_qpage_fts = cnt_qad_fts * 5 + 1 + 1  # + page_ad_num(1) + page_click_num(1)
        fts_index_bias = cnt_user_fts + cnt_qad_fts - 1  # -1: target w/o is_click

        embedding_layer['user'] = self.__feature_embedding(features[:, :cnt_user_fts], 'user')  # [B, 5*D]
        embedding_layer['target'] = self.__ad_embedding(features[:, cnt_user_fts:fts_index_bias],
                                                        is_target=True)  # [B, *D]

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
            ad_embed_seq = torch.stack(ad_embed_seq, 1)  # [B, ad_size, 23*D]
            num_click = features[:, page_index_bias + ad_size * cnt_qad_fts + 1].view(-1)  # [B, ]
            num_click = self.embedding_dict['page_click_num'](num_click)  # [B, D]
            num_click_a = num_click.unsqueeze(1).repeat(1, ad_size, 1)  # [B, A, D]
            ad_embed_seq = torch.cat((ad_embed_seq, num_click_a), dim=2)
            page_seq.append(ad_embed_seq)
            query_embed = self.__query_embedding(features[:, page_index_bias: page_index_bias + query_size])  # [B, D]
            query_embed = torch.cat((query_embed, num_click), dim=1)
            query_seq.append(query_embed)
            num_ad = features[:, page_index_bias + ad_size * cnt_qad_fts].view(-1)
            num_page_ads.append(num_ad)

            for j in range(ad_size):
                ad_features = features[:, page_index_bias + j * cnt_qad_fts: page_index_bias + (j + 1) * cnt_qad_fts]
                mask_nan_ad[:, i, j] = (ad_features[:, -1] == 2)
                mask_click_ad[:, i, j] = (ad_features[:, -1] == 1)

        embedding_layer['page_seq'] = torch.stack(page_seq, 1)  # [B, page_size, ad_size, D]
        embedding_layer['query_seq'] = torch.stack(query_seq, 1)  # [B, page_size, D]
        embedding_layer['num_page_ads'] = torch.stack(num_page_ads, 1).view(batch_size, page_size)  # [B, page_size, ]
        embedding_layer['num_pages'] = features[:, -1].view(-1)
        embedding_layer['mask_nan_ad'] = mask_nan_ad.bool()
        embedding_layer['mask_click_ad'] = mask_click_ad.bool()
        return embedding_layer

    def _page_layer(self, embedding_layer):
        ###  Interest Layer
        page_layer = OrderedDict()
        page_seq = embedding_layer['page_seq']  # [B, P, A, D]
        query_seq = embedding_layer['query_seq']  # [B, P, D]
        target = embedding_layer['target']  # [B, D]
        num_page_ads = embedding_layer['num_page_ads']  # [B, P]
        num_pages = embedding_layer['num_pages']  # [B]
        mask_nan_ad = embedding_layer['mask_nan_ad']  # [B, P, A]
        mask_click_ad = embedding_layer['mask_click_ad']  # [B, P, A]
        device = page_seq.device
        batch_size = page_seq.shape[0]
        page_size = page_seq.shape[1]
        ad_size = page_seq.shape[2]

        # 创建广告级别的掩码
        ad_masks = [torch.arange(ad_size, device=device).view(1, -1) \
                        .repeat(page_size, 1) < num_page_ad.view(-1, 1) \
                    for num_page_ad in num_page_ads]
        page_ad_masks = torch.stack(ad_masks).bool()  # [B, P, A]

        # 创建页面级别的掩码
        page_masks = torch.arange(page_size, device=device).view(1, -1) \
                         .repeat(batch_size, 1) < num_pages.view(-1, 1)
        page_masks = page_masks.bool()

        # 应用广告掩码
        page_seq *= page_ad_masks.float().unsqueeze(-1)  # [B, P, A, D]

        if self.page_layer == 'dynamic_page':
            # 处理广告数为0的页面
            zero_page_ads = (num_page_ads == 0)
            zero_page_ads_masks = zero_page_ads.unsqueeze(2).repeat(1, 1, ad_size)
            mha_page_ad_masks = page_ad_masks | zero_page_ads_masks

            page_rep_list = []
            tmp_target = self.page_net['target_to_adq'](target)  # [B, D]
            current_query = tmp_target

            for ii in range(page_size):
                i = page_size - 1 - ii
                current_page = page_seq[:, i, :, :].squeeze()  # [B, A, D]
                current_page_ad_masks = mha_page_ad_masks[:, i, :].view(batch_size, ad_size)  # [B, A]

                # 计算广告注意力
                current_page_ad_attn = self.page_net['din1'](
                    current_query,
                    current_page,
                    given_mask=current_page_ad_masks)

                # ================ 新增功能开始 ================
                # 获取当前页面的点击和未点击广告掩码
                current_click_mask = mask_click_ad[:, i, :]  # [B, A]
                current_non_click_mask = ~current_click_mask & current_page_ad_masks  # 未点击且有效的广告

                # 计算广告嵌入的相似度矩阵 (余弦相似度)
                normalized_page = F.normalize(current_page, p=2, dim=-1)  # [B, A, D]
                similarity_matrix = torch.matmul(normalized_page, normalized_page.transpose(1, 2))  # [B, A, A]

                # 为每个点击广告计算类内相似度和权重
                click_weights = torch.zeros(batch_size, ad_size, device=device)

                for b in range(batch_size):
                    # 获取当前样本的点击和未点击索引
                    click_indices = torch.where(current_click_mask[b])[0]
                    non_click_indices = torch.where(current_non_click_mask[b])[0]

                    if len(click_indices) == 0 or len(non_click_indices) == 0:
                        continue

                    # 计算每个点击广告与所有未点击广告的相似度
                    for click_idx in click_indices:
                        # 获取与当前点击广告的相似度
                        non_click_sims = similarity_matrix[b, click_idx, non_click_indices]

                        if len(non_click_sims) == 0:
                            continue

                        # 计算平均相似度并筛选高相似度未点击广告
                        avg_sim = non_click_sims.mean()
                        high_sim_mask = non_click_sims > avg_sim
                        high_sim_indices = non_click_indices[high_sim_mask]

                        if len(high_sim_indices) == 0:
                            continue

                        # 构建类内相似度向量 (包括点击广告自身)
                        class_sims = torch.cat([
                            torch.tensor([1.0], device=device),  # 与自身的相似度
                            similarity_matrix[b, click_idx, high_sim_indices]
                        ])

                        # 归一化相似度
                        normalized_sims = F.softmax(class_sims, dim=0)

                        # 计算信息熵作为权重
                        entropy = -torch.sum(normalized_sims * torch.log(normalized_sims + 1e-9))
                        weight = 1.0 / (1.0 + entropy)  # 熵越小权重越大
                        click_weights[b, click_idx] = weight

                # 调整注意力权重：点击广告的权重乘以计算出的权重
                adjusted_attn = current_page_ad_attn.clone()
                adjusted_attn[current_click_mask] *= click_weights[current_click_mask]
                # ================ 新增功能结束 ================

                # 使用调整后的注意力权重
                current_page_rep = (adjusted_attn.view(-1, ad_size, 1) * current_page).sum(1)
                current_page_rep = current_page_rep.view(batch_size, -1)
                page_rep_list.append(current_page_rep)

                # GRU更新查询向量
                gru_input = current_page_rep.view(1, batch_size, -1)
                gru_h0 = current_query.view(1, batch_size, -1)
                output, hn = self.page_net['gru'](gru_input, gru_h0)
                new_query = hn.view(batch_size, -1)
                current_query = new_query

            page_seq_rep = torch.stack(page_rep_list, 1)  # [B, P, D]
            page_query = self.page_net['target_to_pq'](target)  # [B, D]
            page_attn = self.page_net['din2'](page_query, page_seq_rep, num_pages)
            page_rep = (page_attn.view(-1, page_size, 1) * page_seq_rep).sum(1)
            page_rep = page_rep.view(batch_size, -1)

        page_layer['page_rep'] = page_rep
        return page_layer

    def _dnn_layer(self, embedding_layer, page_layer):
        ###  Output Layer
        dnn_layer = OrderedDict()
        mlp_iput = torch.cat([
            embedding_layer['user'],  # 5*embed_dim
            embedding_layer['target'],  # 22*embed_dim
            page_layer['page_rep'],
        ], 1)  # [B, 68*D]
        dnn_layer['dnn_out'] = self.dnn_net['dnn'](mlp_iput)
        return dnn_layer

    def _logits_layer(self, dnn_layer):
        return self.logits_linear(dnn_layer['dnn_out'])

    def forward(self, features, epoch_id=0):
        """
        click_dataset features:
            # user(5), target(31), click_ads(31*N), click_ad_num(1)
        """

        embedding_layer = self._make_embedding_layer(features)
        page_layer = self._page_layer(embedding_layer)
        dnn_layer = self._dnn_layer(embedding_layer, page_layer)
        # dnn_layer = self._dnn_layer(embedding_layer, None)
        logits = self._logits_layer(dnn_layer)
        if epoch_id > 0:
            print('page_ad_attn:', page_layer['page_ad_attn'][0].data.cpu().numpy())
            print('page_attn:', page_layer['page_attn'][0].data.cpu().numpy())

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


