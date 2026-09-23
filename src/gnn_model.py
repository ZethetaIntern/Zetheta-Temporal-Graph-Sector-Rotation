"""
PyTorch Geometric GAT + GRU Architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class TemporalGAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, heads=4):
        super(TemporalGAT, self).__init__()
        self.gat1 = GATConv(in_feats, hidden_feats, heads=heads, dropout=0.2)
        self.gat2 = GATConv(hidden_feats * heads, out_feats, heads=1, dropout=0.2)
        self.gru = nn.GRU(out_feats, out_feats, batch_first=True)

    def forward(self, x, edge_index, h_prev=None):
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gat2(x, edge_index)
        out, h_next = self.gru(x.unsqueeze(0), h_prev)
        return out.squeeze(0), h_next
