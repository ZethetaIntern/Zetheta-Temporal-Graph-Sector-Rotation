import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class TemporalGraphSectorNetwork(nn.Module):
    """
    Institutional-grade architecture combining Graph Attention Networks (GAT)
    and Gated Recurrent Units (GRU) for NSE sectoral index rotation.
    """
    def __init__(self, num_features, hidden_dim, num_classes=1, heads=4):
        super(TemporalGraphSectorNetwork, self).__init__()

        # Spatial Graph Attention Layer
        self.gat1 = GATConv(num_features, hidden_dim, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False)

        # Temporal GRU Layer
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # Output Linear Head for Expected Return / Ranking Prediction
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, h_prev=None):
        """
        x: Node feature tensor [num_nodes, num_features]
        edge_index: Graph connectivity matrix [2, num_edges]
        h_prev: Previous hidden state for temporal recurrence
        """
        # 1. Spatial Graph Convolution Pass
        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.gat2(x, edge_index))

        # Reshape for GRU temporal sequence [batch_size=1, num_nodes, hidden_dim]
        x = x.unsqueeze(0)

        # 2. Temporal Recurrent Pass
        out_gru, h_next = self.gru(x, h_prev)

        # 3. Final Prediction Head
        out = self.fc(out_gru.squeeze(0))
        return out, h_next
