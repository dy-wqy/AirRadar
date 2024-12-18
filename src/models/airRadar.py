import torch
import torch.nn as nn
import torch.nn.functional as F
from src.base.model import BaseModel
from src.layers.embedding import AirEmbedding
import numpy as np
from functools import partial
import sys

from afno.afno1d import AFNO1D

from timm.models.layers import DropPath, to_2tuple, trunc_normal_


dartboard_map = {0: "50-200", 1: "50-200-500", 2: "50", 3: "25-100-250"}


# VAE version
class AirRadar(BaseModel):
    """
    the AirRadar model
    """

    def __init__(
        self,
        dropout=0.3,  # dropout rate
        hidden_channels=32,  # hidden dimension
        mlp_expansion=2,  # the mlp expansion rate in transformers
        num_heads=2,  # the number of heads
        dartboard=0,  # the type of dartboard
        mask_rate=0.5,  # mask token rate
        in_dim=16,  # initial input dim
        context_num=2,
        block_num=2,
        sparsity_threshold=0.01,
        **args,
    ):
        super(AirRadar, self).__init__(**args)
        self.dropout = dropout
        self.skip_convs = nn.ModuleList()
        self.embedding_air = AirEmbedding()
        self._mask_rate = mask_rate
        self.block_num = block_num
        self.get_dartboard_info(dartboard)

        # learnable token for mask nodes
        self.mask_token = nn.Parameter(torch.zeros(1, 1, in_dim))

        # a mlp for converting the input to the embedding
        self.start_mlp = nn.Linear(
            in_features=self.input_dim, out_features=hidden_channels
        )

        self.start_conv = nn.Conv2d(
            in_channels=self.input_dim, out_channels=hidden_channels, kernel_size=(1, 1)
        )
        self.temporal_mlp = nn.Linear(in_features=5, out_features=1)

        self.spatial_embeding = nn.ModuleList()
        for i in range(self.block_num):
            self.spatial_embeding.append(
                Spatial_embedding(
                    hidden_channels=hidden_channels,
                    depth=1,
                    assignment=self.assignment,
                    mlp_expansion=mlp_expansion,
                    num_heads=num_heads,
                    mask=self.mask,
                    dropout=dropout,
                    sparsity_threshold=sparsity_threshold
                )
            )
        self.casual = Casual(
            (hidden_channels + hidden_channels),
            (hidden_channels + hidden_channels),
            (hidden_channels + hidden_channels),
            context_num,
            in_dim,
        )
        # self.out_mlp = nn.Linear((hidden_channels + hidden_channels), in_dim)

    def initialize_weights(self):
        torch.nn.init.normal_(self.mask_token, std=0.02)

    def get_dartboard_info(self, dartboard):
        """
        get dartboard-related attributes
        """
        path_assignment = (
            "data/local_partition/" + dartboard_map[dartboard] + "/assignment.npy"
        )
        path_mask = "data/local_partition/" + dartboard_map[dartboard] + "/mask.npy"
        print(path_assignment)
        self.assignment = (
            torch.from_numpy(np.load(path_assignment)).float().to(self.device)
        )
        self.mask = torch.from_numpy(np.load(path_mask)).bool().to(self.device)

    def encoding_mask_noise(self, x, mask_ratio=0.75, mask_nodes=None):
        B, N, D = x.shape  # batch, length, dim

        if mask_nodes is not None:
            mask = torch.ones([B, N], device=x.device)
            mask_nodes = torch.tensor(mask_nodes, device=x.device)
            mask[:, mask_nodes] = 0
            x_keeped = torch.mul(x, mask.unsqueeze(-1))
            mask_tokens = self.mask_token.expand(B, 1, D)
            out_x = x_keeped + (1 - mask.unsqueeze(-1)) * mask_tokens
        else:
            len_keep = int(N * (1 - mask_ratio))

            noise = torch.rand(B, N, device=x.device)  # noise in [0, 1]

            # sort noise for each sample

            ids_shuffle = torch.argsort(
                noise, dim=1
            )  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            # keep the first subset
            ids_keep = ids_shuffle[:, :len_keep]
            x_keeped = torch.gather(
                x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
            )

            # generate the binary mask: 1 is keep, 0 is masked
            mask = torch.zeros([B, N], device=x.device)
            mask[:, :len_keep] = 1
            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)

            mask_tokens = self.mask_token.repeat(
                x_keeped.shape[0], ids_restore.shape[1] + 1 - x_keeped.shape[1], 1
            )
            out_x = torch.cat([x_keeped, mask_tokens], dim=1)
            out_x = torch.gather(
                out_x,
                dim=1,
                index=ids_restore.unsqueeze(-1).repeat(1, 1, out_x.shape[2]),
            )  # unshuffle)

        return out_x, mask

    def forward(self, inputs, history, mask_nodes=None, pred_attr="PM25", g=None):
        """
        inputs: the historical data
        supports: adjacency matrix (actually our method doesn't use it)
                 Including adj here is for consistency with GNN-based methods
        """
        # Inputs: [b,n,c]
        x_masked, mask = self.encoding_mask_noise(inputs, self._mask_rate, mask_nodes)
        history = history * mask.unsqueeze(2).unsqueeze(3)
        x_embed = self.embedding_air(x_masked[..., 11:15].long())
        temporal_embed = self.embedding_air(history[..., 11:15].long())
        x_temporal = torch.cat(
            (history[..., :11], temporal_embed, history[..., 15:]), -1
        )  # [b, n, t ,c]
        x_spatial = torch.cat(
            (x_masked[..., :11], x_embed, x_masked[..., 15:]), -1
        )  # [b, n, c]

        x_temporal = self.start_conv(x_temporal.permute(0, 3, 2, 1)).permute(
            0, 3, 1, 2
        )  # [b, n , c, t]
        x_temporal = self.temporal_mlp(x_temporal).squeeze(-1)
        x_spatial = self.start_mlp(x_spatial)

       
        x = torch.cat([x_spatial, x_temporal], dim=-1)

        for i in range(self.block_num):
            x = self.spatial_embeding[i](x)
        x = self.casual(x)
       
        if pred_attr == "PM25":
            x = x[..., 0].unsqueeze(-1)  # PM2.5
       
        return x, mask


# For both encoder and decoder
class Spatial_embedding(nn.Module):
    def __init__(
        self,
        hidden_channels,
        depth,
        assignment,
        mlp_expansion,
        num_heads,
        mask,
        dropout,
        sparsity_threshold,
    ):
        super().__init__()

        self.local_coder = DS_MSA(
            hidden_channels * 2,
            depth=1,
            heads=num_heads,
            mlp_dim=hidden_channels * mlp_expansion,
            assignment=assignment,
            mask=mask,
            dropout=dropout,
        )

        self.global_coder = FftNet(hidden_channels * 2,sparsity_threshold=sparsity_threshold,depth=1)

        self.output_layer = nn.Linear(
            (hidden_channels + hidden_channels) * 2, (hidden_channels + hidden_channels)
        )

    def forward(self, x):
        local_x = self.local_coder(x)
        global_x = self.global_coder(x)
        # x = global_x
        x = torch.cat([local_x, global_x], dim=-1)
        out = self.output_layer(x)

        return out


class SpatialAttention(nn.Module):
    # dartboard project + MSA
    def __init__(
        self,
        dim,
        heads=4,
        qkv_bias=False,
        qk_scale=None,
        dropout=0.0,
        num_sectors=17,
        assignment=None,
        mask=None,
    ):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."

        self.dim = dim
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim**-0.5
        self.num_sector = num_sectors
        self.assignment = assignment  # [n, n, num_sector]
        self.mask = mask  # [n, num_sector]

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_linear = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.relative_bias = nn.Parameter(torch.randn(heads, 1, num_sectors))
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [b, n, c]

        B, N, C = x.shape

        # query: [bn, 1, c]
        # key/value target: [bn, num_sector, c]
        # [b, n, num_sector, c]
        pre_kv = torch.einsum("bnc,mnr->bmrc", x, self.assignment)

        pre_kv = pre_kv.reshape(-1, self.num_sector, C)  # [bn, num_sector, c]
        pre_q = x.reshape(-1, 1, C)  # [bn, 1, c]

        q = (
            self.q_linear(pre_q)
            .reshape(B * N, -1, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )  # [bn, num_heads, 1, c//num_heads]
        kv = (
            self.kv_linear(pre_kv)
            .reshape(B * N, -1, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]  # [bn, num_heads, num_sector, c//num_heads]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = (
            attn.reshape(B, N, self.num_heads, 1, self.num_sector) + self.relative_bias
        )  # you can fuse external factors here as well
        mask = self.mask.reshape(1, N, 1, 1, self.num_sector)

        # masking
        attn = (
            attn.masked_fill_(mask, float("-inf"))
            .reshape(B * N, self.num_heads, 1, self.num_sector)
            .softmax(dim=-1)
        )

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DS_MSA(nn.Module):
    # Dartboard Spatial MSA
    def __init__(
        self,
        dim,  # hidden dimension
        depth,  # number of MSA in DS-MSA
        heads,  # number of heads
        mlp_dim,  # mlp dimension
        assignment,  # dartboard assignment matrix
        mask,  # mask
        dropout=0.0,
    ):  # dropout rate
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        SpatialAttention(
                            dim,
                            heads=heads,
                            dropout=dropout,
                            assignment=assignment,
                            mask=mask,
                            num_sectors=assignment.shape[-1],
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        # x: [b, c, n, t]
        b, n, c = x.shape
        # x = x + self.pos_embedding  # [b*t, n, c]  we use relative PE instead
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        # x = x.reshape(b, t, n, c).permute(0, 3, 2, 1)
        return x


# Pre Normalization in Transformer
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# FFN in Transformer
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sparsity_threshold=0.01,
        use_fno=False,
        use_blocks=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.filter = AFNO1D(
            hidden_size=dim,
            num_blocks=1,
            sparsity_threshold=sparsity_threshold,
            hard_thresholding_fraction=1,
            hidden_size_factor=1,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.double_skip = True

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x


class FftNet(nn.Module):
    def __init__(
        self,
        embed_dim=32,
        depth=2,
        mlp_ratio=4.0,
        representation_size=None,
        uniform_drop=False,
        drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        dropcls=0,
        sparsity_threshold=0.01,
        use_fno=False,
        use_blocks=False,
    ):
        """
        Args:
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.pos_embed = nn.Parameter(torch.zeros(1, 1085, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        if uniform_drop:
            print("using uniform droppath with expect rate", drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            print("using linear droppath with expect rate", drop_path_rate * 0.5)
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, depth)
            ]  # stochastic depth decay rule
        # dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    sparsity_threshold = sparsity_threshold,
                    use_fno=use_fno,
                    use_blocks=use_blocks,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, N, C = x.shape

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x


class Casual(nn.Module):
    def __init__(
        self, input_dim, expert_hidden_dim, expert_output_dim, num_experts, output_dim
    ):
        super(Casual, self).__init__()
        self.experts = nn.ModuleList(
            [
                Expert(input_dim, expert_hidden_dim, expert_output_dim)
                for _ in range(num_experts)
            ]
        )
        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, num_experts), nn.Softmax(dim=1)
        )
        self.out_dim = output_dim
        self.output_layer = nn.Linear(expert_output_dim, output_dim)

    def forward(self, x):
        b, n, c = x.shape
        x = x.reshape(b * n, c)
        gate = self.gating_network(x)  # dynamic weight
        expert_outputs = [expert(x) for expert in self.experts]

        weighted_expert_outputs = [
            gate[:, i].unsqueeze(1) * expert_outputs[i]
            for i in range(len(self.experts))
        ]
        final_output = torch.sum(torch.stack(weighted_expert_outputs, dim=2), dim=2)

        return self.output_layer(final_output).reshape(b, n, self.out_dim)


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
