



import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender


class SimpleTransformer(nn.Module):


    def __init__(self, d_model, nhead=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # src: (batch, seq_len, d_model)
        src2 = self.self_attn(src, src, src, need_weights=False)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src





class LGMRec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(LGMRec, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.cf_model = 'lightgcn'
        self.n_mm_layer = config['n_mm_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.n_hyper_layer = config['n_hyper_layer']
        self.hyper_num = config['hyper_num']
        self.keep_rate = config['keep_rate']
        self.alpha = config['alpha']
        self.cl_weight = config['cl_weight']
        self.reg_weight = config['reg_weight']
        self.tau = 0.4

        self.n_nodes = self.n_users + self.n_items
        self.hgnnLayer = HGNNLayer(self.n_hyper_layer)
        self.num_samples = 20
        self.transformer = None
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(
            np.float32)


        self.adj = self.scipy_matrix_to_sparse_tenser(self.interaction_matrix, torch.Size(
            (self.n_users, self.n_items)))
        self.num_inters, self.norm_adj = self.get_norm_adj_mat()
        self.num_inters = torch.FloatTensor(1.0 / (self.num_inters + 1e-7)).to(
            self.device)


        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.drop = nn.Dropout(p=1 - self.keep_rate)


        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=True)
            self.item_image_trs = nn.Parameter(nn.init.xavier_uniform_(
                torch.zeros(self.v_feat.shape[1], self.feat_embed_dim)))
            self.v_hyper = nn.Parameter(nn.init.xavier_uniform_(
                torch.zeros(self.v_feat.shape[1], self.hyper_num)))
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=True)
            self.item_text_trs = nn.Parameter(
                nn.init.xavier_uniform_(torch.zeros(self.t_feat.shape[1], self.feat_embed_dim)))
            self.t_hyper = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.t_feat.shape[1], self.hyper_num)))


        if self.v_feat is not None and self.t_feat is not None:
            self.proj_cge = nn.Linear(self.embedding_dim, self.feat_embed_dim)
            self.transformer = SimpleTransformer(
                d_model=self.feat_embed_dim,
                nhead=4,
                dim_feedforward=self.feat_embed_dim * 4,
                dropout=0.1
            )


            self.policy_mlp = nn.Sequential(
                nn.Linear(2 * self.feat_embed_dim, self.feat_embed_dim),
                nn.ReLU(),
                nn.Linear(self.feat_embed_dim, 1)
            )

        self.num_candidates = 9
        self.num_local = 3
        self.num_sim = 3
        self.num_random = 3
        self.rl_k = 3

        self.neighbors = [[] for _ in range(self.n_nodes)]

        for u, i in zip(self.interaction_matrix.row, self.interaction_matrix.col):
            self.neighbors[u].append(i + self.n_users)

        for u, i in zip(self.interaction_matrix.row, self.interaction_matrix.col):
            self.neighbors[i + self.n_users].append(u)

        self.neighbors_tensor = [torch.tensor(nei, dtype=torch.long, device=self.device) for nei in self.neighbors]

        max_neighbors = max(len(nei) for nei in self.neighbors)
        self.neighbors_padded = torch.full((self.n_nodes, max_neighbors), -1, dtype=torch.long, device=self.device)
        for i, nei in enumerate(self.neighbors):
            if nei:
                self.neighbors_padded[i, :len(nei)] = torch.tensor(nei, dtype=torch.long, device=self.device)


        self.log_probs = None


    def scipy_matrix_to_sparse_tenser(self, matrix, shape):
        row = matrix.row
        col = matrix.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(matrix.data)
        return torch.sparse.FloatTensor(i, data, shape).to(self.device)



    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)

        sumArr = (A > 0).sum(axis=1)

        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D


        L = sp.coo_matrix(L)
        return sumArr, self.scipy_matrix_to_sparse_tenser(L, torch.Size(
            (self.n_nodes, self.n_nodes)))


    def cge(self):
        if self.cf_model == 'mf':
            cge_embs = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        if self.cf_model == 'lightgcn':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight),
                                       dim=0)
            cge_embs = [ego_embeddings]
            for _ in range(self.n_ui_layers):
                ego_embeddings = torch.sparse.mm(self.norm_adj,
                                                 ego_embeddings)
                cge_embs += [ego_embeddings]
            cge_embs = torch.stack(cge_embs, dim=1)
            cge_embs = cge_embs.mean(dim=1, keepdim=False)
        return cge_embs


    def mge(self, str='v'):
        if str == 'v':
            item_feats = torch.mm(self.image_embedding.weight, self.item_image_trs)
        elif str == 't':
            item_feats = torch.mm(self.text_embedding.weight, self.item_text_trs)
        user_feats = torch.sparse.mm(self.adj, item_feats) * self.num_inters[:self.n_users]
        mge_feats = torch.cat([user_feats, item_feats], dim=0)
        for _ in range(self.n_mm_layer):
            mge_feats = torch.sparse.mm(self.norm_adj, mge_feats)
        return mge_feats



    def sample_k_without_replacement(self, probs, k):

        selected_idx = torch.multinomial(probs, k, replacement=False)

        selected_probs = probs.gather(1, selected_idx)

        log_probs = torch.log(selected_probs + 1e-10).sum(dim=1)

        return selected_idx, log_probs

    def forward(self):

        if self.v_feat is not None:
            iv_hyper = torch.mm(self.image_embedding.weight, self.v_hyper)
            uv_hyper = torch.mm(self.adj, iv_hyper)
            iv_hyper = F.gumbel_softmax(iv_hyper, self.tau, dim=1, hard=False)
            uv_hyper = F.gumbel_softmax(uv_hyper, self.tau, dim=1, hard=False)
        if self.t_feat is not None:
            it_hyper = torch.mm(self.text_embedding.weight, self.t_hyper)
            ut_hyper = torch.mm(self.adj, it_hyper)
            it_hyper = F.gumbel_softmax(it_hyper, self.tau, dim=1, hard=False)
            ut_hyper = F.gumbel_softmax(ut_hyper, self.tau, dim=1, hard=False)

        cge_embs = self.cge()

        if self.v_feat is not None and self.t_feat is not None:

            v_feats = self.mge('v')
            t_feats = self.mge('t')
            self.cge_proj = self.proj_cge(cge_embs)
            self.v_feats = F.normalize(v_feats)
            self.t_feats = F.normalize(t_feats)


            self.lge_embs = cge_embs + self.v_feats + self.t_feats
            n_nodes = self.n_nodes
            device = self.device

            neigh_part = self.neighbors_padded[:, :self.num_local]

            rand_fill = torch.randint(0, n_nodes, (n_nodes, self.num_local), device=device)

            valid_mask = (neigh_part != -1)
            neigh_part = torch.where(valid_mask, neigh_part, rand_fill)
            neigh_part = neigh_part[:, :2]
            norm_emb = F.normalize(self.lge_embs).half()
            norm_emb_t = norm_emb.t().contiguous()

            chunk_size = 4096

            sim_topk_list = []

            for start in range(0, n_nodes, chunk_size):
                end = min(start + chunk_size, n_nodes)

                chunk = norm_emb[start:end]

                sim_chunk = torch.mm(chunk.half(), norm_emb_t)

                diag_idx = torch.arange(end - start, device=device)

                sim_chunk[diag_idx, start + diag_idx] = -1e4

                topk = torch.topk(
                    sim_chunk,
                    self.num_sim,
                    dim=1
                ).indices

                sim_topk_list.append(topk)

            sim_topk = torch.cat(sim_topk_list, dim=0)

            sim_part = sim_topk[:, :2]


            random_part = torch.randint(0, n_nodes, (n_nodes, 2), device=device)


            candidates = torch.cat([neigh_part, sim_part, random_part], dim=1)


            self_emb_expand = self.lge_embs[:, None, :]
            cand_emb = self.lge_embs[candidates]
            concat = torch.cat([self_emb_expand.expand_as(cand_emb), cand_emb], dim=-1)
            scores = self.policy_mlp(concat).squeeze(-1)
            probs = F.softmax(scores, dim=-1)


            selected_idx, log_probs = self.sample_k_without_replacement(probs,
                                                                        self.rl_k)


            batch_idx = torch.arange(n_nodes, device=device).unsqueeze(1).expand(-1, self.rl_k)
            selected_cand_emb = cand_emb[batch_idx, selected_idx]
            self_emb = self.lge_embs.unsqueeze(1)
            seq = torch.cat([self_emb, selected_cand_emb], dim=1)
            seq_out = self.transformer(seq)

            all_embs = seq_out[:, 0, :]
            self.log_probs = log_probs

            self.alpha1 =
            self.final_embs1 = self.lge_embs + self.alpha1 * F.normalize(all_embs)

            uv_hyper_embs, iv_hyper_embs = self.hgnnLayer(self.drop(iv_hyper), self.drop(uv_hyper), cge_embs[self.n_users:])
            ut_hyper_embs, it_hyper_embs = self.hgnnLayer(self.drop(it_hyper), self.drop(ut_hyper), cge_embs[self.n_users:])
            av_hyper_embs = torch.cat([uv_hyper_embs, iv_hyper_embs], dim=0)
            at_hyper_embs = torch.cat([ut_hyper_embs, it_hyper_embs], dim=0)
            ghe_embs = av_hyper_embs + at_hyper_embs
            self.final_embs = self.final_embs1 + self.alpha * F.normalize(ghe_embs)

        else:
            self.lge_embs = None
            self.final_embs = cge_embs
            self.log_probs = None
        u_final, i_final = torch.split(self.final_embs, [self.n_users, self.n_items], dim=0)
        return u_final, i_final, self.log_probs, [uv_hyper_embs, iv_hyper_embs, ut_hyper_embs, it_hyper_embs]


    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        return bpr_loss


    def ssl_triple_loss(self, emb1, emb2, all_emb):
        norm_emb1 = F.normalize(emb1)
        norm_emb2 = F.normalize(emb2)
        norm_all_emb = F.normalize(all_emb)


        pos_score = torch.exp(torch.mul(norm_emb1, norm_emb2).sum(dim=1) / self.tau)
        ttl_score = torch.exp(torch.matmul(norm_emb1, norm_all_emb.T) / self.tau).sum(dim=1)
        ssl_loss = -torch.log(pos_score / ttl_score).sum()
        return ssl_loss



    def contrast_pair_loss(self, emb_a, emb_b, all_b):
        norm_emb_a = F.normalize(emb_a)
        norm_emb_b = F.normalize(emb_b)
        norm_all_b = F.normalize(all_b)

        pos_score = torch.exp(torch.mul(norm_emb_a, norm_emb_b).sum(dim=1) / self.tau)
        ttl_score = torch.exp(torch.matmul(norm_emb_a, norm_all_b.T) / self.tau).sum(dim=1)
        loss = -torch.log(pos_score / ttl_score).sum()
        return loss


    def reg_loss(self, *embs):
        reg_loss = 0
        for emb in embs:
            reg_loss += torch.norm(emb, p=2)
        reg_loss /= embs[-1].shape[0]
        return reg_loss

    def calculate_loss(self, interaction):
        ua_embeddings, ia_embeddings, log_probs, hyper_embeddings = self.forward()

        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]


        batch_bpr_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        batch_contrast_loss = 0

        batch_reg_loss = self.reg_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        if self.v_feat is not None and self.t_feat is not None:

            batch_node_indices = torch.cat([users, pos_items + self.n_users])

            batch_cge_proj = self.cge_proj[batch_node_indices]
            batch_v_feats = self.v_feats[batch_node_indices]
            batch_t_feats = self.t_feats[batch_node_indices]


            loss_cge_v = self.contrast_pair_loss(batch_cge_proj, batch_v_feats, self.v_feats)
            loss_cge_t = self.contrast_pair_loss(batch_cge_proj, batch_t_feats, self.t_feats)
            loss_v_t = self.contrast_pair_loss(batch_v_feats, batch_t_feats, self.t_feats)

            self.cvtcl_weight =

            batch_contrast_loss = (loss_v_t + loss_cge_v + loss_cge_t) * self.cvtcl_weight


        [uv_embs, iv_embs, ut_embs, it_embs] = hyper_embeddings
        batch_hcl_loss = self.ssl_triple_loss(uv_embs[users], ut_embs[users], ut_embs) + self.ssl_triple_loss(
            iv_embs[pos_items], it_embs[pos_items], it_embs)

        rl_loss = 0.0
        if log_probs is not None:
            pos_scores = (u_g_embeddings * pos_i_g_embeddings).sum(dim=1)
            neg_scores = (u_g_embeddings * neg_i_g_embeddings).sum(dim=1)
            reward_user = pos_scores - neg_scores
            reward_pos = pos_scores - neg_scores
            reward_neg = neg_scores - pos_scores


            user_log_probs = log_probs[users]
            pos_log_probs = log_probs[pos_items]
            neg_log_probs = log_probs[neg_items]


            self.rl_weight =
            rl_loss = - (reward_user.detach() * user_log_probs +
                         reward_pos.detach() * pos_log_probs +
                         reward_neg.detach() * neg_log_probs).mean() * self.rl_weight

        loss = batch_bpr_loss + self.cl_weight * batch_hcl_loss + batch_contrast_loss  + self.reg_weight * batch_reg_loss + rl_loss

        return loss


    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embs, item_embs,log_probs, hyper_embeddings = self.forward()
        scores = torch.matmul(user_embs[user], item_embs.T)
        return scores

class HGNNLayer(nn.Module):
    def __init__(self, n_hyper_layer):
        super(HGNNLayer, self).__init__()
        self.h_layer = n_hyper_layer


    def forward(self, i_hyper, u_hyper, embeds):
        i_ret = embeds
        for _ in range(self.h_layer):
            lat = torch.mm(i_hyper.T, i_ret)
            i_ret = torch.mm(i_hyper, lat)
            u_ret = torch.mm(u_hyper, lat)
        return u_ret, i_ret











