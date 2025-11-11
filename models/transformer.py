import math
from typing import Tuple, Type

import torch
from torch import Tensor
from torch import nn

from .mlp import MLP


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        num_layers: int,
        emb_dim: int,
        num_heads: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
        norm: bool,
    ):

        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                emb_dim, num_heads, dropout, layer_norm_eps,
                mlp_factor, norm_first, activation
            ) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(emb_dim, layer_norm_eps) if norm else nn.Identity()

    def forward(self, src, pos_emb, src_mask, src_key_padding_mask):
        output = src
        for layer in self.layers:
            output = layer(output, pos_emb, src_mask, src_key_padding_mask)
        return self.norm(output)


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
    ):
        super(TransformerEncoderLayer, self).__init__()

        self.norm_first = norm_first

        self.norm1 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.norm2 = nn.LayerNorm(emb_dim, layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.self_attn = nn.MultiheadAttention(
            emb_dim, num_heads, dropout
        )
        self.mlp = MLP(emb_dim, mlp_factor * emb_dim, dropout, activation)

    def with_emb(self, x, emb):
        return x if emb is None else x + emb

    def forward(self, src, pos_emb, src_mask, src_key_padding_mask):
        if self.norm_first:
            src_norm = self.norm1(src)
            q = k = src_norm + pos_emb
            src = src + self.dropout1(self.self_attn(
                query=q,
                key=k,
                value=src_norm,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask
            )[0])

            src_norm = self.norm2(src)
            src = src + self.dropout2(self.mlp(src_norm))
        else:
            q = k = src + pos_emb
            src = self.norm1(src + self.dropout1(self.self_attn(
                query=q,
                key=k,
                value=src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask
            )[0]))

            src = self.norm2(src + self.dropout2(self.mlp(src)))

        return src





class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                CrossAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                query=queries,
                keys=keys,
            )

        return keys


class TransformerAdapt(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                AttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_downsample_rate=attention_downsample_rate,
                )
            )


    def forward(
        self,
        adapted_image_embedding: Tensor,
        image_pe: Tensor,
        image_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = adapted_image_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=image_embedding,
                query_pe=image_pe,
                key_pe=image_pe,
            )

        return queries

class AttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn = Attention(
            embedding_dim, num_heads)
        self.norm2 = nn.LayerNorm(embedding_dim)


    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:


        queries = self.self_attn(q=queries, k=queries, v=queries)
        queries = self.norm1(queries)

        # Cross attention block
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        return queries, keys


class SelfCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
    ) -> None:
        """
        """
        super().__init__()
        self.self_attention =  Attention(embedding_dim, num_heads)
        self.cross_attention =  Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(
        self, image_f: Tensor, adapted_image_f: Tensor, pos_enc: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        adapted_image_f  = adapted_image_f+ self.self_attention(q=adapted_image_f+pos_enc,
                                         k=adapted_image_f+pos_enc,
                                         v=adapted_image_f+pos_enc)
        adapted_image_f = self.norm1(adapted_image_f)
        adapted_image_f = adapted_image_f + self.cross_attention(q=adapted_image_f+pos_enc,
                                          k=image_f+pos_enc,
                                          v=image_f+pos_enc)
        adapted_image_f = self.norm2(adapted_image_f)
        return adapted_image_f

class PrototypeAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
    ) -> None:
        """
        """
        super().__init__()
        self.cross_attention =  Attention(embedding_dim, num_heads)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(
        self, image_f: Tensor, prototypes: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        image_f = image_f + self.cross_attention(q=image_f,
                                          k=prototypes,
                                          v=prototypes)
        image_f = self.norm(image_f)
        return image_f



class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out
