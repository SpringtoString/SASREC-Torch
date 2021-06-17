import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class SAS(nn.Module):
    def __init__(self,n_users,n_items,max_len=10,embedding_size=4,  l2_reg_embedding=0.00001, n_blocks=2, n_heads=1,dropout_rate=0.0):
        super().__init__()
        self.model_name = 'SAS'
        self.n_items = n_items + 1
        self.embedding_size = embedding_size
        self.l2_reg_embedding = l2_reg_embedding
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate

        self.embedding_dict = nn.ModuleDict({
            'item_emb': self.create_embedding_matrix(self.n_items, embedding_size),
            'postion_emb': self.create_embedding_matrix(max_len, embedding_size),
        })

        ''' SA network '''
        self.attn_normlayers = nn.ModuleList([nn.LayerNorm(embedding_size, eps=1e-8) for i in range(n_blocks)])
        self.att_layers = nn.ModuleList([nn.MultiheadAttention(embed_dim=embedding_size, num_heads=n_heads) for i in range(n_blocks)])
        self.forward_normlayers = nn.ModuleList([nn.LayerNorm(embedding_size, eps=1e-8) for i in range(n_blocks)])
        self.forward_layers = nn.ModuleList([PointWiseFeedForward(embedding_size, dropout_rate) for i in range(n_blocks)])
        self.last_layernorm = torch.nn.LayerNorm(embedding_size, eps=1e-8)

        self.device = 'cpu'    # 这个值由 trainer设置改变

    def log2feats(self, log_seqs):
        # log_seqs [B,T]
        seqs = self.embedding_dict['item_emb'](log_seqs) # shape is [B,T,E]
        seqs *= self.embedding_size ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1]) # [B,T]
        seqs += self.embedding_dict['postion_emb'](torch.LongTensor(positions).to(self.device))
        # seqs = self.emb_dropout(seqs)

        timeline_mask = (log_seqs == 0) # [B,T]

        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim *mask,item 0都置为0向量了

        tl = seqs.shape[1] # time dim len for enforce causality T=max_len

        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.device))  # [T, T] 上三角矩阵

        for i in range(self.n_blocks):
            seqs = torch.transpose(seqs, 0, 1)         # [T, B, E]
            Q = self.attn_normlayers[i](seqs)     # layer_norm
            mha_outputs, _ = self.att_layers[i](Q, seqs, seqs,
                                            attn_mask=attention_mask)
                                            # key_padding_mask=timeline_mask
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)        # [B, T, E]

            seqs = self.forward_normlayers[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, T, C)

        return log_feats

    def computeL2LOSS(self,seqs):
        seqs_vector = self.embedding_dict['item_emb'](seqs)
        emb_loss = torch.norm(seqs_vector) ** 2
        return emb_loss

    def forward(self, input_dict):
        '''

        :param input_dict:
        :return:   rui, ruj
        '''
        users, seqs, pos_items, neg_items = input_dict['users'],input_dict['seqs'], input_dict['pos_items'], input_dict['neg_items']   # [B,T]

        sas_vector = self.log2feats(seqs)              # [B, T, E]
        pos_items_vector = self.embedding_dict['item_emb'](pos_items)    # [B, T, E]
        neg_items_vector = self.embedding_dict['item_emb'](neg_items)    # [B, T, E]

        ''' compute click logit '''
        pos_logits = (sas_vector * pos_items_vector).sum(dim=-1)    # [B, T] 感觉这里要加sigmoid
        neg_logits = (sas_vector * neg_items_vector).sum(dim=-1)    # [B, T]
        loss = self.sq_bce_loss(seqs, pos_logits, neg_logits)

        ''' compute sequential loss '''
        emb_loss = self.computeL2LOSS(seqs)
        emb_loss = self.l2_reg_embedding * emb_loss

        return loss, emb_loss

    def sq_bce_loss(self, seqs, pos_logits, neg_logits):
        bce_criterion = torch.nn.BCEWithLogitsLoss()   # 注意这里感觉为logits加上 sigmoid
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.device), torch.zeros(neg_logits.shape,device=self.device)
        indices = torch.where(seqs != 0)
        loss = bce_criterion(pos_logits[indices], pos_labels[indices])
        loss += bce_criterion(neg_logits[indices], neg_labels[indices])
        return loss

    def create_embedding_matrix(self, vocabulary_size, embedding_size, init_std=0.0001, sparse=False,):
        embedding = nn.Embedding(vocabulary_size, embedding_size, sparse=sparse)
        nn.init.normal_(embedding.weight, mean=0, std=init_std)
        return embedding

    def rating(self, log_seqs, all_item): # for inference
        log_feats = self.log2feats(log_seqs) # user_ids hasn't been used yet  [B, T, E]

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.embedding_dict['item_emb'](all_item) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)
