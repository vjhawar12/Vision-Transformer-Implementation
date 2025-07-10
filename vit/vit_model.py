import torch
import torch.nn as nn
from math import sqrt
from timm.models.layers import DropPath

# image pre-processing helper class

class PatchEmbedding(nn.Module):

    def __init__(self, batches=32, in_channels=3, patch_size=16, size=128, embed_dim=768):
        super().__init__()

        assert size % patch_size == 0, "Image size must be divisible by patch size"

        self.batches = batches
        self.in_channels = in_channels # rgb ==> 3 channels
        self.patch_size = patch_size # size of each patch (like a token)
        self.embed_dim = embed_dim # the higher-dimensional space to project the patches to
        self.size = size # size of input image
        self.N = (self.size // self.patch_size) ** 2 # number of patches

        self.proj = nn.Conv2d(in_channels=self.in_channels, # B, C, H, W --> B, D, H_p, W_p
                              out_channels=self.embed_dim, # 3D space --> 768D space to extract more information
                              kernel_size=self.patch_size, # so that the patches don't overlap
                              stride=self.patch_size) # divides input image into patches

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim)) # token which captures the 'meaning' of the image in a vector
        self.pos_embeddings = nn.Parameter(torch.randn(1, self.N + 1, self.embed_dim)) # learnable positional embeddings

    def forward(self, x):
        x = self.proj(x) # applying conv2d projection
        x = torch.flatten(x, 2) # B, D, N
        x = x.transpose(1, 2) # B, N, D
        B = x.shape[0]
        cls_token = self.cls_token.expand(B, -1, -1) # expanding the cls token along the batch dimension so it can be added later
        x = torch.cat((cls_token, x), dim=1) # adding the cls token to the input tensor
        x = x + self.pos_embeddings # adding the positional embeddings to the input tensor

        return x
    

# A head has unique Q, K, V matrices and computes attention

class ManualMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=768, heads=12, dropout=0.15):
        super().__init__()

        assert embed_dim % heads == 0, "Embedding dimension must be divisible by heads"

        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads

        # fully connected NN layers with # of input neurons = embed_dim = # output neurons
        self.Q_proj = nn.Linear(embed_dim, embed_dim)
        self.V_proj = nn.Linear(embed_dim, embed_dim)
        self.K_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        self.output = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, D = x.shape

        # Q: what it's looking for
        # K: the kind of features it has
        # V: the actual representation

        # sending the input tensor through proj NN layers
        Q = self.Q_proj(x) # batches, patches, embed_dim
        Q = Q.view(B, N, self.heads, self.head_dim).permute(0, 2, 1, 3) # single head --> multihead

        V = self.V_proj(x) # batches, patches, embed_dim
        V = V.view(B, N, self.heads, self.head_dim).permute(0, 2, 1, 3) # single head --> multihead

        K = self.K_proj(x) # batches, patches, embed_dim
        K = K.view(B, N, self.heads, self.head_dim).permute(0, 2, 1, 3) # single head --> multihead

        # computing attention
        x = self.compute_attention(Q, K, V).permute(0, 2, 1, 3).contiguous() # B, heads, N, head_dim --> B, N, heads, head_dim
        x = self.dropout(x)
        x = x.view(B, N, D)
        x = self.output(x)

        return x

    def compute_attention(self, Q, K, V):
        K_T = torch.transpose(K, -2, -1) # transpose so that multiplication is defined
        scaling = sqrt(self.head_dim)
        val = torch.matmul(Q, K_T) / scaling

        return torch.matmul(torch.softmax(val, dim=-1), V)


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, heads=12, dropout=0.15):
        super().__init__()

        self.mhsa = ManualMultiHeadSelfAttention(embed_dim, heads)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.drop_path_1 = DropPath(0.05)
        self.drop_path_2 = DropPath(0.05)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.ln1(x) # normalizing activation functions to prevent saturation. Pre-norm is expected to lead to faster convergence.
        attn_out = self.mhsa(x) # computing residual
        x = x + self.drop_path_1(attn_out) # skip connection
        x = self.ln2(x) # normalizing activation functions to prevent saturation
        ffn_out = self.ffn(x) # computing residual
        x = x + self.drop_path_2(ffn_out) # skip connection

        return x


class VisionTransformer(nn.Module):
    def __init__(self,  batches=32, in_channels=3, patch_size=16, size=128, embed_dim=768, heads=12, depth=8, num_classes=10, dropout=0.15):
        super().__init__()

        self.patch_embedding = PatchEmbedding(batches, in_channels, patch_size, size, embed_dim)

        # each transformer applies attention through various heads -- each with unique Q, K, and V matrices
        self.transformer_stack = nn.ModuleList(TransformerEncoder(embed_dim, heads) for _ in range(depth))

        self.mlp_head = nn.Sequential( # maps cls token to logits
            nn.Dropout(dropout),
            nn.Linear(in_features=embed_dim, out_features=num_classes),
        )

    def forward(self, x):
        x = self.patch_embedding(x)

        for t in self.transformer_stack:
            x = t(x)

        cls = x[:, 0]
        x = self.mlp_head(cls)

        return x
    
