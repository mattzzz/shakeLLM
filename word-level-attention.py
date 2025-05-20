import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load and lowercase the text
with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

# Use Keras tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
word_index = tokenizer.word_index
index_word = tokenizer.index_word
vocab_size = len(word_index) + 1  # +1 for padding

# Convert full text to sequence of word indices
tokens = tokenizer.texts_to_sequences([text])[0]

seq_length = 10
sequences = []

for i in range(seq_length, len(tokens)):
    seq = tokens[i - seq_length:i + 1]
    sequences.append(seq)

sequences = np.array(sequences)
X, y = sequences[:, :-1], sequences[:, -1]


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

embed_dim = 64
num_heads = 2
ff_dim = 128
max_len = seq_length

inputs = Input(shape=(seq_length,))
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embed_dim)(inputs)
x = PositionalEncoding(max_len, embed_dim)(embedding_layer)
x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
x = GlobalAveragePooling1D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(vocab_size, activation='softmax')(x)

model = Model(inputs, outputs)
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import Adam

from tensorflow.keras.metrics import TopKCategoricalAccuracy

optimizer = Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy', TopKCategoricalAccuracy(k=5)])

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


model.fit(X, y, batch_size=128, epochs=15)


import random

def generate_text(seed_text, next_words=50):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=seq_length, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)[0]
        predicted = np.random.choice(len(predicted_probs), p=predicted_probs)
        seed_text += ' ' + index_word.get(predicted, '')
    return seed_text


print(generate_text("to be or not to be", next_words=50))



