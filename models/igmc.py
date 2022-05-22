import torch
from torch.nn import Linear, LSTM, Embedding
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import dropout_adj


class IGMC(torch.nn.Module):
    def __init__(
            self, dataset, gconv=RGCNConv,
            latent_dim=[32, 32, 32, 32], num_relations=5,
            num_bases=4, side_features=False,
            n_side_features=0, adj_dropout=0.2,
    ):
        super(IGMC, self).__init__()
        self.adj_dropout = adj_dropout
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            gconv(
                dataset.num_features, latent_dim[0],
                num_relations, num_bases,
            ),
        )
        for i in range(0, len(latent_dim) - 1):
            self.convs.append(
                gconv(
                    latent_dim[i], latent_dim[i + 1],
                    num_relations, num_bases,
                ),
            )

        # self.emb = Embedding(90524,3233)
        self.rnn1 = LSTM(2 * sum(latent_dim),2 * sum(latent_dim),num_layers=2)
        self.lin1 = Linear(2 * sum(latent_dim), 128)
        self.side_features = side_features
        if side_features:
            self.lin1 = Linear(2 * sum(latent_dim) + n_side_features, 128)
            self.rnn1 = LSTM(2 * sum(latent_dim) + n_side_features,2 * sum(latent_dim) + n_side_features,num_layers=2)

        self.lin2 = Linear(128, 1)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        for i in self.convs:
            i.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch
        # print(x)
        # print(x.shape)
        # break
        if self.adj_dropout > 0:
            edge_index, edge_type = dropout_adj(
                edge_index,
                edge_type,
                p=self.adj_dropout,
                num_nodes=len(x),
                training=self.training,
            )
        # print(batch)
        # print(sum(latent_dim))
        # hx = torch.randn(2, batch, 2 * sum(latent_dim))
        # cx = torch.randn(2, batch, 2 * sum(latent_dim))
        concat_states = []
        # print(x.shape)
        # print(edge_index.shape)
        # print(edge_type.shape)
        # x = self.emb(x.long())
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index, edge_type))
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)


        users = data.x[:, 0] == 1
        items = data.x[:, 1] == 1
        x = torch.cat([concat_states[users], concat_states[items]], 1)

        if self.side_features:
            x = torch.cat([x, data.u_feature, data.v_feature], 1)
        # print(x.shape)
        x,hidden = self.rnn1(x)
        # print(x.shape)
        # print(hidden.shape)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.adj_dropout, training=self.training)
        x = self.lin2(x)

        return x[:, 0]
