from transformers.models.roformer.modeling_roformer import RoFormerEncoder, RoFormerConfig
import torch
import torch.nn as nn


class TransformerModel(torch.nn.Module):

    def __init__(self, hidden_size, num_hidden_layers, num_attention_heads, intermediate_size, max_seq_len) -> None:
        super().__init__()
        self.model = RoFormerEncoder(
            RoFormerConfig(
                vocab_size=2,  # not using it
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                max_position_embeddings=max_seq_len
            )
        )
    
    def forward(self, hidden_states, hidden_mask):
        # hidden_states [batch, len, dim]
        # hidden_mask [batch, len]
        full_mask = (1.0 - hidden_mask[:, None, None, :]) * torch.finfo(torch.float).min if hidden_mask is not None else None
        out = self.model(
            hidden_states=hidden_states,
            attention_mask=full_mask
        ).last_hidden_state
        return out


class CrossLayerV2(nn.Module):
    """
    DCN-v2 Cross Layer

    x_{l+1} = x0 ⊙ (W xl + b) + xl

    如果 low_rank 不为 None，则使用低秩分解：
        W = U V
    """

    def __init__(self, input_dim: int, low_rank: int = 32):
        super().__init__()

        self.low_rank = low_rank

        if low_rank is None:
            self.transform = nn.Linear(input_dim, input_dim)
        else:
            self.down = nn.Linear(input_dim, low_rank, bias=False)
            self.up = nn.Linear(low_rank, input_dim)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        """
        x0: 原始输入
        xl: 当前 cross layer 输入
        shape: [batch, input_dim]
        """

        if self.low_rank is None:
            wx = self.transform(xl)
        else:
            wx = self.up(self.down(xl))

        return x0 * wx + xl


class CrossNetwork(nn.Module):
    """
    统一使用 CrossLayerV2 的 Cross Network。
    """

    def __init__(
        self,
        input_dim: int,
        num_cross_layers: int = 3,
        low_rank: int = 32,
    ):
        super().__init__()

        self.cross_layers = nn.ModuleList([
            CrossLayerV2(input_dim, low_rank=low_rank)
            for _ in range(num_cross_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, input_dim]
        """

        x0 = x
        xl = x

        for layer in self.cross_layers:
            xl = layer(x0, xl)

        return xl


class MLP(nn.Module):
    """
    Deep Network。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims=(256, 128),
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []

        last_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            last_dim = hidden_dim

        if layers:
            self.mlp = nn.Sequential(*layers)
        else:
            self.mlp = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DCN(nn.Module):
    """
    Deep & Cross Network，统一使用 CrossLayerV2。

    Input:
        x: [batch, feature_num]

    Output:
        logits: [batch, output_dim]
    """

    def __init__(
        self,
        feature_num: int,
        hidden_dims=(128, 128, 64),
        num_cross_layers: int = 3,
        low_rank: int = 32,
        output_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.feature_num = feature_num

        self.cross_network = CrossNetwork(
            input_dim=feature_num,
            num_cross_layers=num_cross_layers,
            low_rank=low_rank,
        )

        self.deep_network = MLP(
            input_dim=feature_num,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        if hidden_dims:
            deep_output_dim = hidden_dims[-1]
        else:
            deep_output_dim = feature_num

        self.output_layer = nn.Linear(
            in_features=feature_num + deep_output_dim,
            out_features=output_dim,
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, feature_num]
        """

        cross_out = self.cross_network(x)
        deep_out = self.deep_network(x)

        concat_out = torch.cat([cross_out, deep_out], dim=-1)

        logits = self.output_layer(concat_out)

        return logits


if __name__ == "__main__":
    m = DCN(64)
    print(m(torch.randn([4,64])))
    # m = TransformerModel(32, 4, 4, 256)
    # mask = torch.randint(0,2,[4, 10])
    # print((m(torch.randn([4, 10, 32]), torch.randint(0,1,[4, 10]))*mask.unsqueeze(-1).float()))