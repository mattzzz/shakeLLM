import numpy as np

# Load text
with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

# Create character vocabulary
chars = sorted(list(set(text)))
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

# Encode entire text
encoded = np.array([char2idx[c] for c in text])

# Create input-output pairs
seq_length = 100
X = []
y = []

for i in range(0, len(encoded) - seq_length):
    X.append(encoded[i:i + seq_length])
    y.append(encoded[i + seq_length])

X = np.array(X)
y = np.array(y)


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, LayerNormalization, Dropout, Layer
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


embed_dim = 64
num_heads = 2
ff_dim = 128

inputs = Input(shape=(seq_length,))
x = Embedding(vocab_size, embed_dim)(inputs)
x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(vocab_size, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.summary()


model.fit(X, y, batch_size=64, epochs=10)


def generate_text(model, seed, length=200):
    result = seed.lower()
    for _ in range(length):
        input_seq = [char2idx.get(c, 0) for c in result[-seq_length:]]
        input_seq = np.pad(input_seq, (seq_length - len(input_seq), 0))
        input_seq = np.expand_dims(input_seq, axis=0)
        pred = model.predict(input_seq, verbose=0)[0]
        next_idx = np.random.choice(len(pred), p=pred)
        result += idx2char[next_idx]
    return result

print(generate_text(model, "to be or not to be,"))


