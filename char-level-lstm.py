# Read in text
with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()

# Create character vocabulary
chars = sorted(list(set(text)))
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

# Convert text to integers
encoded_text = [char2idx[c] for c in text]

# Sequence length
seq_length = 100
step = 1
sequences = []
next_chars = []

for i in range(0, len(encoded_text) - seq_length, step):
    sequences.append(encoded_text[i: i + seq_length])
    next_chars.append(encoded_text[i + seq_length])

import numpy as np
X = np.array(sequences)
y = np.array(next_chars)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

model = Sequential([
    Embedding(vocab_size, 64, input_length=seq_length),
    LSTM(128, return_sequences=False),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
model.summary()


model.fit(X, y, batch_size=128, epochs=10)


def generate_text(seed, length=200):
    result = seed.lower()
    for _ in range(length):
        input_seq = [char2idx.get(c, 0) for c in result[-seq_length:]]
        input_seq = np.expand_dims(input_seq, axis=0)
        pred = model.predict(input_seq, verbose=0)[0]
        next_idx = np.random.choice(len(pred), p=pred)
        result += idx2char[next_idx]
    return result

print(generate_text("to be or not to be"))


