import sentencepiece as spm
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization, Dense, Dropout, Embedding, Input
from tensorflow.keras.models import Model

# Reuse the BPE tokenizer trained by subword-attention.py
# Run subword-attention.py first if shakespeare_bpe.model doesn't exist
sp = spm.SentencePieceProcessor()
sp.load('shakespeare_bpe.model')

with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

token_ids = sp.encode(text, out_type=int)

seq_length = 64
sequences = []
for i in range(seq_length, len(token_ids)):
    seq = token_ids[i - seq_length:i + 1]
    sequences.append(seq)

sequences = np.array(sequences)
X, y = sequences[:, :-1], sequences[:, -1]
vocab_size = sp.get_piece_size()


class CausalMultiHeadSelfAttention(Layer):
    """Multi-head self-attention with a causal (upper-triangular) mask.

    Each position can only attend to itself and earlier positions, matching
    the autoregressive behaviour of GPT at both train and inference time.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = embed_dim // num_heads
        self.wq = Dense(embed_dim)
        self.wk = Dense(embed_dim)
        self.wv = Dense(embed_dim)
        self.dense = Dense(embed_dim)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        q = self.split_heads(self.wq(x), batch_size)
        k = self.split_heads(self.wk(x), batch_size)
        v = self.split_heads(self.wv(x), batch_size)

        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.depth, tf.float32))

        # Upper-triangular mask — add -1e9 to future positions so softmax zeros them out
        causal_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        scores += causal_mask[tf.newaxis, tf.newaxis, :, :] * -1e9

        weights = tf.nn.softmax(scores, axis=-1)
        output = tf.matmul(weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat = tf.reshape(output, (batch_size, -1, self.embed_dim))
        return self.dense(concat)


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.att = CausalMultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="gelu"),  # GELU used in GPT-2
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x, training=False):
        attn_output = self.att(x)
        out1 = self.layernorm1(x + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output, training=training))


embed_dim = 128
num_heads = 4
ff_dim = 512
num_layers = 4

inputs = Input(shape=(seq_length,))

# Learned positional embeddings (GPT-style) rather than fixed sinusoidal
token_emb = Embedding(vocab_size, embed_dim)(inputs)
pos_emb = Embedding(seq_length, embed_dim)(tf.range(seq_length))
x = token_emb + pos_emb

for _ in range(num_layers):
    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)

x = LayerNormalization(epsilon=1e-6)(x)

# Use the last token's representation — with causal masking it has attended to
# the full prefix. This is how every decoder-only LLM produces next-token predictions.
# Previous models used GlobalAveragePooling1D which collapsed sequence order.
x = x[:, -1, :]

outputs = Dense(vocab_size, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss='sparse_categorical_crossentropy'
)
model.summary()

model.fit(X, y, batch_size=64, epochs=5)


def generate_text(seed, next_tokens=100, temperature=0.8):
    """Generate text with temperature scaling.

    temperature < 1.0 — more focused/conservative
    temperature > 1.0 — more random/creative
    """
    ids = sp.encode(seed.lower(), out_type=int)
    for _ in range(next_tokens):
        input_seq = ids[-seq_length:]
        input_seq = np.pad(input_seq, (seq_length - len(input_seq), 0))
        input_seq = np.expand_dims(input_seq, axis=0)
        pred_probs = model.predict(input_seq, verbose=0)[0].astype(np.float64)
        # Temperature scaling: divide log-probs before re-normalising
        pred_probs = np.exp(np.log(pred_probs + 1e-10) / temperature)
        pred_probs /= pred_probs.sum()
        next_id = np.random.choice(len(pred_probs), p=pred_probs)
        ids.append(next_id)
    return sp.decode(ids)


print(generate_text("To be, or not to be", next_tokens=100))
