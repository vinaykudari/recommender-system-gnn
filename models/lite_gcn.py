import torch
from torch import nn
from torch_sparse import matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing


class LightGCN(MessagePassing):
    def __init__(
            self, num_users, num_items,
            emb_dim=64, num_layers=3, self_loop=False,
            user_emb=None, item_emb=None,
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.self_loop = self_loop

        self.user_emb = nn.Embedding(
            num_embeddings=self.num_users,
            embedding_dim=self.emb_dim,
        )
        self.item_emb = nn.Embedding(
            num_embeddings=self.num_items,
            embedding_dim=self.emb_dim,
        )

        # experiment: use custom (pretrained) embeddings or initialize using normal distribution
        if user_emb:
            assert user_emb.shape == (self.num_users, self.emb_dim), f'Expected dim {self.num_users, self.emb_dim}'
            self.user_emb.weight.data.copy_(torch.from_numpy(user_emb))
        else:
            nn.init.normal_(self.user_emb.weight, std=0.1)

        if item_emb:
            assert user_emb.shape == (self.num_items, self.emb_dim), f'Expected dim {self.num_items, self.emb_dim}'
            self.item_emb.weight.data.copy_(torch.from_numpy(item_emb))
        else:
            nn.init.normal_(self.item_emb.weight, std=0.1)

    def forward(self, edge_index, **kwargs):
        # experiment: use ratings as edge weights?

        edge_index_norm = gcn_norm(
            edge_index,
            add_self_loops=self.self_loop,
        )
        emb_0 = torch.cat(
            [
                self.user_emb.weight,
                self.item_emb.weight,
            ],
        )
        embs = [emb_0]
        emb_k = emb_0

        # multi-scale diffusion
        for i in range(self.num_layers):
            emb_k = self.propagate(edge_index_norm, x=emb_k)
            embs.append(emb_k)

        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1)

        user_emb, item_emb = torch.split(
            emb_final, [self.num_users, self.num_items],
        )

        return user_emb, self.user_emb.weight, item_emb, self.item_emb.weight

    def message(self, x_j):
        return x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x)
