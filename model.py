import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer
import math
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv
from torch.nn import Sequential
from torch.nn import Linear, Sequential, BatchNorm1d, LeakyReLU
from torch_geometric.nn import global_add_pool



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=50000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)

        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class GIN(torch.nn.Module):

    def __init__(self, hidden_channels, num_layers, gin_dim_feedforward):
        super(GIN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(GINConv(
            Sequential(Linear(hidden_channels, hidden_channels*gin_dim_feedforward),
                       BatchNorm1d(hidden_channels*gin_dim_feedforward), LeakyReLU(),
                       Linear(hidden_channels*gin_dim_feedforward, hidden_channels*gin_dim_feedforward), LeakyReLU())))
        self.bns.append(BatchNorm1d(hidden_channels*gin_dim_feedforward))

        for _ in range(1, num_layers):
            self.convs.append(GINConv(
                Sequential(Linear(hidden_channels*gin_dim_feedforward, hidden_channels*gin_dim_feedforward),
                           BatchNorm1d(hidden_channels*gin_dim_feedforward), LeakyReLU(),
                           Linear(hidden_channels*gin_dim_feedforward, hidden_channels*gin_dim_feedforward), LeakyReLU())))
            self.bns.append(BatchNorm1d(hidden_channels*gin_dim_feedforward))

        self.lin1 = Linear(hidden_channels*gin_dim_feedforward * num_layers+64 , 64)
        self.bn1 = BatchNorm1d(64)
        self.lin2 = Linear( 64 , 2)

    def forward(self, x, edge_index, batch, kmers):
        h_list = []

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.leaky_relu(x)
            h_list.append(global_add_pool(x, batch))
        h = torch.cat(h_list, dim=1)
        h_list_concatenated = torch.cat(h_list, dim=1)
        h = torch.cat([h_list_concatenated, kmers], dim=1)
        h = h.float()
        h = self.lin1(h)
        h = F.leaky_relu(h)
        h = self.bn1(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        return h, F.log_softmax(h, dim=1)
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.activation1 = nn.LeakyReLU(negative_slope=0.01)


    def forward(self, src , src_mask = None, src_key_padding_mask = None):
        src2,attention_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation1(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src,attention_weights

class TransGIN(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes, num_heads, atten_num_layers, atten_dim_feedforward,gin_num_layers,gin_dim_feedforward,dropout_rate = 0.2):
        super(TransGIN, self).__init__()
        self.embedding1 = nn.Embedding(num_node_features, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        self.gin_model = GIN(hidden_dim,gin_num_layers,gin_dim_feedforward)
        self.transformer_encoder = []
        for i in range(atten_num_layers):
            self.transformer_encoder.append(TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*atten_dim_feedforward, dropout = 0.1))
        self.transformer_encoder= nn.ModuleList(self.transformer_encoder)



    def forward(self, src0, threshold):
        length = src0.size(0)
        src1 = src0[:,:64]
        src1 = src1.long()
        kmers = src0[:,64:128]
        sequence1 = self.embedding1(src1)
        sequence1_swapped = sequence1.permute(1, 0, 2)
        sequence1_swapped = self.positional_encoding(sequence1_swapped)
        sequence1_swapped = sequence1_swapped.float()
        attention_weights_all=[]
        for layer in self.transformer_encoder:
            sequence1_swapped,attention_weights_layer=layer(sequence1_swapped)
            attention_weights_all.append(attention_weights_layer)
        sequence1_swapped = sequence1_swapped.permute(1, 0, 2)

        graphs = []
        for d in range(src1.size(0)):
            edges_list = []
            for attention_weights in attention_weights_all:
                attention_weights_d = attention_weights[d]
                mask = (attention_weights_d > threshold).float()
                edge_index = torch.nonzero(mask, as_tuple=True)
                edge_index = torch.stack(edge_index, dim=0)
                edges_list.append(edge_index)
            if len(edges_list) > 1:
                edges_concat = torch.cat(edges_list, dim=1)
            else:
                edges_concat = edges_list[0] if edges_list else None
            reversed_edge_index = torch.flip(edges_concat, [0])
            unique_edge = torch.unique(torch.cat([edges_concat, reversed_edge_index], dim=1), dim=1)
            x1 = sequence1_swapped[d]
            graph1 = Data(x=x1, edge_index=unique_edge.contiguous())
            graphs.append(graph1)

        loader = DataLoader(graphs, batch_size=length, shuffle=False)

        for graph in loader:
            gat_features,a = self.gin_model(graph.x, graph.edge_index, graph.batch,kmers)

        output = gat_features

        return output


            
