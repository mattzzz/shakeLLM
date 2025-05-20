import sentencepiece as spm

# Save Shakespeare text temporarily
with open("shakespeare.txt", "r", encoding="utf-8") as f:
    with open("sp_text.txt", "w", encoding="utf-8") as out:
        out.write(f.read().lower())

# Train SentencePiece model
spm.SentencePieceTrainer.train(input='sp_text.txt', model_prefix='shakespeare_bpe', vocab_size=8000)

sp = spm.SentencePieceProcessor()
sp.load('shakespeare_bpe.model')

# Encode/decode text
ids = sp.encode("To be, or not to be", out_type=int)
tokens = sp.encode("To be, or not to be", out_type=str)
print("Token IDs:", ids)
print("Subwords:", tokens)


# Tokenize full corpus
with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()
token_ids = sp.encode(text, out_type=int)

# Create training sequences
seq_length = 32
sequences = []
for i in range(seq_length, len(token_ids)):
    seq = token_ids[i - seq_length:i + 1]
    sequences.append(seq)

import numpy as np
sequences = np.array(sequences)
X, y = sequences[:, :-1], sequences[:, -1]
vocab_size = sp.get_piece_size()



from tensorflow.keras.layers import Layer, LayerNormalization
import numpy as np
import tensorflow as tf

class MultiHeadSelfAttention(Layer):
    def __init__(self, embed_dim, num_heads=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0

        self.depth = embed_dim // num_heads
        self.wq = Dense(embed_dim)
        self.wk = Dense(embed_dim)
        self.wv = Dense(embed_dim)
        self.dense = Dense(embed_dim)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]

        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.depth, tf.float32))
        weights = tf.nn.softmax(scores, axis=-1)
        output = tf.matmul(weights, v)

        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat = tf.reshape(output, (batch_size, -1, self.embed_dim))
        return self.dense(concat)

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x):
        attn_output = self.att(x, x, x)
        out1 = self.layernorm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output))
    

class PositionalEncoding(Layer):
    def __init__(self, max_len, embed_dim):
        super().__init__()
        pos_enc = np.zeros((max_len, embed_dim))
        for pos in range(max_len):
            for i in range(0, embed_dim, 2):
                pos_enc[pos, i] = np.sin(pos / (10000 ** ((2 * i)/embed_dim)))
                if i + 1 < embed_dim:
                    pos_enc[pos, i+1] = np.cos(pos / (10000 ** ((2 * (i+1))/embed_dim)))
        self.pos_enc = tf.constant(pos_enc[np.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        return x + self.pos_enc[:, :tf.shape(x)[1], :]


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, GlobalAveragePooling1D, Dropout
# Same TransformerBlock & PositionalEncoding from before...

inputs = Input(shape=(seq_length,))
x = Embedding(input_dim=vocab_size, output_dim=64)(inputs)
x = PositionalEncoding(seq_length, 64)(x)
x = TransformerBlock(embed_dim=64, num_heads=2, ff_dim=128)(x)
x = GlobalAveragePooling1D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.1)(x)
outputs = Dense(vocab_size, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.summary()

# Train
model.fit(X, y, batch_size=64, epochs=5)


def generate_bpe_text(seed, next_tokens=50):
    ids = sp.encode(seed.lower(), out_type=int)
    for _ in range(next_tokens):
        input_seq = ids[-seq_length:]
        input_seq = np.pad(input_seq, (seq_length - len(input_seq), 0))
        input_seq = np.expand_dims(input_seq, axis=0)
        pred_probs = model.predict(input_seq, verbose=0)[0]
        next_id = np.random.choice(len(pred_probs), p=pred_probs)
        ids.append(next_id)
    return sp.decode(ids)

print(generate_bpe_text("To be, or not to be", next_tokens=50))


