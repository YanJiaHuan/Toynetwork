'''
This is a transformer written in flax 0.7.2, jax 0.4.13, jaxlib 0.4.13
'''
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from typing import Any
import flax,jaxlib,jax

print("Flax version:", flax.__version__)
print("JAX version:", jax.__version__)
print("JAXlib version:", jaxlib.__version__)

#+-----------------------------------------------------------------------------+
### self attention ###
class SelfAttention(nn.Module):
    embed_size = int
    heads = int

    @nn.compact
    def __call__(self, x, mask):
        head_dim = self.embed_size // self.heads
        assert head_dim * self.heads == self.embed_size, "Embedding size needs to be divisible by number of heads"

        queries = nn.Dense(features=self.embed_size)(x)
        Keys = nn.Dense(features=self.embed_size)(x)
        Values = nn.Dense(features=self.embed_size)(x)

        # reshape into (batch_size, sequence length, heads, head_dim)
        queries = queries.reshape(queries.shape[0], queries.shape[1], self.heads, head_dim)
        keys = Keys.reshape(Keys.shape[0], Keys.shape[1], self.heads, head_dim)
        values = Values.reshape(Values.shape[0], Values.shape[1], self.heads, head_dim)

        # calculate energy score
        energy = jnp.einsum("nqhd,nkhd->nhqk", queries, keys) # shape (batch_size, heads, query_len, key_len)
        if mask is not None:
            energy =  jnp.where(mask == 0, float("-1e20"), energy)
        # calculate attention score
        attention = nn.softmax(energy/jnp.sqrt(head_dim),axis=-1)
        # apply attention weight to value
        out = jnp.einsum("nhqk,nkhd->nqhd", attention, values)
        # Concatenate the heads and pass through a final linear layer
        out = out.reshape(out.shape[0], out.shape[1], self.embed_size)
        out = nn.Dense(features=self.embed_size)(out)

        return out

class TransformerBlock(nn.Module):
    embed_size = int
    heads = int
    forward_expansion = int
    dropout = float

    @nn.compact
    def __call__(self, x, mask):
        attention = SelfAttention(self.embed_size, self.heads)(x, mask)
        x = nn.LayerNorm()(attention + x)  # Add residual connection and normalize
        x = nn.Dropout(rate=self.dropout)(x, deterministic=True)  # Dropout during training

        forward = nn.Dense(features=self.forward_expansion * self.embed_size)(x)
        forward = nn.relu(forward)
        forward = nn.Dense(features=self.embed_size)(forward)
        out = nn.LayerNorm()(forward + x)  # Add residual connection and normalize
        out = nn.Dropout(rate=self.dropout)(out, deterministic=True)  # Dropout during training
        return out
