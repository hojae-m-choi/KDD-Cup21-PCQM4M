from functools import wraps
from math import pi, log

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn, einsum


# helpers
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cache_fn(f):
    cache = None

    @wraps(f)
    def cached_fn(*args, _cache=True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn


# helper classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


# main class
class Perceiver(nn.Module):
    def __init__(
            self,
            *,
            depth,
            input_channels,
            self_per_cross_attn=1,
            num_latents=512,
            latent_dim=512,
            cross_heads=1,
            latent_heads=8,
            cross_dim_head=64,
            latent_dim_head=64,
            attn_dropout=0.,
            ff_dropout=0.,
            weight_tie_layers=False,
    ):
        super().__init__()
        input_axis = 1
        self.input_axis = input_axis
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        get_cross_attn = lambda: \
            PreNorm(
                latent_dim,
                Attention(
                    latent_dim,
                    input_channels,
                    heads=cross_heads,
                    dim_head=cross_dim_head,
                    dropout=attn_dropout
                ),
                context_dim=input_channels
            )
        get_cross_ff = lambda: \
            PreNorm(
                latent_dim,
                FeedForward(
                    latent_dim,
                    dropout=ff_dropout
                )
            )
        get_latent_attn = lambda: \
            PreNorm(
                latent_dim,
                Attention(
                    latent_dim,
                    heads=latent_heads,
                    dim_head=latent_dim_head,
                    dropout=attn_dropout,
                )
            )
        get_latent_ff = lambda: \
            PreNorm(
                latent_dim,
                FeedForward(
                    latent_dim,
                    dropout=ff_dropout,
                )
            )

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(
            cache_fn,
            (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff)
        )

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for _ in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args),
                    get_latent_ff(**cache_args)
                ]))

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns
            ]))

        self.regression = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 1)
        )

    def forward(self, data, mask=None):
        b, *axis, _, device = *data.shape, data.device
        assert len(axis) == self.input_axis, 'input data must have the right number of axis'

        # concat to channels of data and flatten axis
        data = rearrange(data, 'b ... d -> b (...) d')

        x = repeat(self.latents, 'n d -> b n d', b=b)

        # layers
        for cross_attn, cross_ff, self_attns in self.layers:
            x = cross_attn(x, context=data, mask=mask) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x

        x = x.mean(dim=-2)
        return self.regression(x)
