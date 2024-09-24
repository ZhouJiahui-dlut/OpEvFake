import torch
import sys
from torch import nn
import torch.nn.functional as F

from modules.transformer import TransformerEncoder


class MULTModel(nn.Module):
    def __init__(self, orig_d_l, orig_d_a, orig_d_v, MULT_d, mult_dropout):
        """
        Construct a MulT model.
        """
        super(MULTModel, self).__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v = orig_d_l, orig_d_a, orig_d_v
        self.d_l, self.d_a, self.d_v = MULT_d, MULT_d, MULT_d
        self.num_heads = 2
        self.layers = 5
        self.attn_dropout = 0.1
        self.attn_dropout_a = 0.0
        self.attn_dropout_v = 0.0
        self.relu_dropout = 0.1
        self.res_dropout = 0.1
        self.out_dropout = mult_dropout
        self.embed_dropout = 0.25
        self.attn_mask = True

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.proj_g = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_l_with_v = self.get_network(self_type='lv')
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_a_with_v = self.get_network(self_type='av')
        self.trans_v_with_a = self.get_network(self_type='va')

        self.trans_g_with_l = self.get_network(self_type='l')
        self.trans_g_with_a = self.get_network(self_type='la')
        self.trans_g_with_v = self.get_network(self_type='lv')

        self.trans_l_with_g = self.get_network(self_type='l')
        self.trans_a_with_g = self.get_network(self_type='la')
        self.trans_v_with_g = self.get_network(self_type='lv')
       
        '''# Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)'''

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask,
                                  position_emb = True)
            
    def forward(self, x_l, x_g, x_a, x_v):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_l = F.dropout(x_l.transpose(2, 1), p=self.embed_dropout, training=self.training)
        x_g = x_g.transpose(2, 1)
        x_a = x_a.transpose(2, 1)
        x_v = x_v.transpose(2, 1)
       
        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_g = x_g if self.orig_d_l == self.d_l else self.proj_g(x_g)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_g = proj_x_g.permute(2, 0, 1)

        # (A,V,G) --> L
        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)  # Dimension (L, N, d_l)
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
        h_l_with_gs = self.trans_l_with_g(proj_x_l, proj_x_g, proj_x_g)
        h_ls = F.dropout(torch.cat([h_l_with_as, h_l_with_vs, h_l_with_gs], dim=2), p=self.out_dropout, training=self.training)
        # (L,V,G) --> A
        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
        h_a_with_gs = self.trans_a_with_g(proj_x_a, proj_x_g, proj_x_g)
        h_as = F.dropout(torch.cat([h_a_with_ls, h_a_with_vs, h_a_with_gs], dim=2), p=self.out_dropout, training=self.training)
        # (L,A,G) --> V
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
        h_v_with_gs = self.trans_v_with_g(proj_x_v, proj_x_g, proj_x_g)
        h_vs = F.dropout(torch.cat([h_v_with_ls, h_v_with_as, h_v_with_gs], dim=2), p=self.out_dropout, training=self.training)

        # (L,A,V) --> G
        h_g_with_ls = self.trans_g_with_l(proj_x_g, proj_x_l, proj_x_l)
        h_g_with_as = self.trans_g_with_a(proj_x_g, proj_x_a, proj_x_a)
        h_g_with_vs = self.trans_g_with_v(proj_x_g, proj_x_g, proj_x_g)
        h_gs = F.dropout(torch.cat([h_g_with_ls, h_g_with_as, h_g_with_vs], dim=2), p=self.out_dropout, training=self.training)

        return h_ls, h_gs, h_as, h_vs

