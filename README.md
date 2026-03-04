# ShakeLLM: A Study of Language Model Architectures Using Shakespeare's Works

This project implements and compares various text generation models trained on Shakespeare's works, exploring different approaches from character-level to word-level modeling with modern attention mechanisms. It serves as both a practical study of language model architectures and a demonstration of how different approaches perform on a rich, structured literary corpus.

## Why Shakespeare?

Shakespeare's works are an ideal dataset for studying language models because:
- Rich vocabulary and complex language structures
- Consistent writing style across a large corpus
- Mix of poetry and prose
- Well-defined character voices and dialogue
- Historical significance and cultural relevance
- Public domain availability
- Structured format (sonnets, plays, etc.)
- Challenging but learnable patterns

The complete works of Shakespeare used in this project can be found at [Project Gutenberg](https://www.gutenberg.org/ebooks/100) (The Complete Works of William Shakespeare).


## Usage

First download the complete works of Shakespeare from Project Gutenberg to shakespeare.txt
Each model can be run independently. For example, to run the character-level LSTM model:

```bash
python char-level-lstm.py
```

The models will train on the Shakespeare dataset and can generate new text given a seed phrase.


## Project Overview

The project contains several implementations of text generation models, ordered from simplest to most complex:

### **Character-Level LSTM** (`char-level-lstm.py`)
   - Simple LSTM-based model operating at the character level
   - Uses embedding layer followed by LSTM and dense layers
   - Good for understanding basic sequence modeling
   - Baseline implementation with simplest architecture
   - Expected to perform fourth due to basic architecture
   - ~112k params
   - total training time on Apple M3 Pro ~1.3h


```bash
42021/42021 [==============================] - 464s 11ms/step - loss: 1.4358
to be or not to bedinezeruse hermase, heal all some here’er of lost to biddy, the head. master
each whether us, and then being his own samides of each poid,
well store them to pantly, you’v’lled her, and they, that wer
```

### **Character-Level Attention** (`char-level-attention.py`)
   - Attention-based model operating at the character level
   - Combines the benefits of attention with character-level modeling
   - More flexible but computationally expensive
   - Third in expected performance
   - Good intermediate step between LSTM and transformer models
   - ~56k params
   - total training time on GTX 1080 ~2h

```bash
84041/84041 [==============================] - 881s 10ms/step - loss: 3.0128
to be or not to be,ethteitetloiiacnn iai re np hoy i et ts   rn  t hne win tln, ftulaic e lhn ir 
 oea eny x tf,htpkyfha,m ’ctcyis1oeiwei4d0,to hho dp
a dif,fhss nne tuoicnnr;;irnrbeweallcrrais r en wakn heerfof flrne f
```

### **Word-Level Attention** (`word-level-attention.py`)
   - Implements a transformer-based architecture
   - Uses multi-head self-attention mechanism
   - Includes positional encoding and transformer blocks
   - More sophisticated approach for word-level generation
   - ~5.7M params
   - total training time on GTX 1080 ~30mins

```bash
$ python word-level-attention.py 

7599/7599 [==============================] - 120s 16ms/step - loss: 5.7222 - accuracy: 0.1095 
to be or not to be blessing to be with thine and his ancient traitor a maid some other loss himself may i are well neither here set thy highness’ spite of thy just plot he will have one benefit let’s show your pure blazon him truly there can sing with good on give me and
```

### **Subword Attention** (`subword-attention.py`)
   - Similar to word-level attention but operates on subword units
   - Uses BPE (Byte Pair Encoding) for tokenization
   - Good balance between character and word-level modeling
   - Expected best performance expected due to efficient tokenization
   - Modern approach to handling vocabulary
   - ~1.5M params
   - total training time on GTX 1080 ~15mins


```bash
$ python subword-attention.py

Epoch 5/5
20768/20768 [==============================] - 165s 8ms/step - loss: 4.8897
to be, or not to be to your lordship upon a man? a widow. the brother the shallow lords to music, the thieves in patience, i are sick, calling forlorn will as can approach this man. , this, but so most earth is over my beautyuous
```

### **GPT Decoder** (`gpt-decoder.py`)
   - Decoder-only transformer — the architecture behind GPT
   - Fixes the three key flaws in the earlier attention models (see analysis below)
   - BPE tokenization (reuses the tokenizer trained in `subword-attention.py`)
   - 4 stacked transformer layers vs 1 in previous models
   - Learned positional embeddings instead of sinusoidal
   - GELU activation (used in GPT-2/3) instead of ReLU
   - Temperature-controlled generation
   - ~3M params
   - total training time on GTX 1080 ~55mins

```bash
$ python gpt-decoder.py

Epoch 5/5
20768/20768 [==============================] - 679s 33ms/step - loss: 4.1711
to be, or not to be music from from when it is, the but then did, although, how to make for pyramus call to sing but such to be but plain fair she did then the story of this from the place of the other witness of this deep disgrace, for there, and in these creatures where their love is too sweet, and all their love is not old, in many giddy bidding. but to the sight of all that glory for the earth. what shall i think? that is not
```

---

## Architecture Analysis

### Why Each Model Performed How It Did

**char-level-lstm** achieved the lowest *numerical* loss (1.43), but loss values aren't comparable across models because character-level vocab (~70 tokens) is much smaller than word or subword vocab. The output produces real words but breaks down quickly — LSTM struggles to hold coherent context beyond ~50 characters and has no mechanism to relate distant tokens.

**char-level-attention** produced the worst output (essentially noise). Two structural problems:
1. `GlobalAveragePooling1D` collapses the entire sequence into one vector, discarding all positional information — the attention computation is then wasted.
2. No causal masking means the model attends to future characters during training, but generates one character at a time at inference — a mismatch between train and test regimes.

**word-level-attention** produced the most readable output despite a high-looking loss — the loss is high because it's predicting over a ~25k word vocabulary. Positional encoding and transformer blocks help significantly. Limited by a tiny context window (10 words) and still suffers from `GlobalAveragePooling1D` and no causal masking.

**subword-attention** produced the best output. BPE tokenization handles punctuation and rare words efficiently, `seq_length=32` covers more semantic context than 10 words, and training is fast. Has the same structural flaws as the word-level model but benefits more from its tokenization.

### The Three Key Gaps vs a Real LLM

| Issue | Earlier attention models | GPT Decoder |
|---|---|---|
| **Causal masking** | No — attends to future tokens during training | Yes — upper-triangular mask blocks future positions |
| **Sequence output** | `GlobalAveragePooling1D` collapses to one vector | Last-token representation preserves sequential context |
| **Depth** | 1 transformer layer | 4 stacked transformer layers |

**Causal masking** is the most important fix. Without it, the model learns a different task during training (fill in the middle) than it performs at inference (predict the next token), so the learned representations don't transfer cleanly.

**Last-token selection** (`x[:, -1, :]`) uses the final position's representation — which, with causal masking, has attended to the entire prefix but nothing beyond. This is how GPT and every modern decoder-only LLM produces next-token predictions.

**Depth** is where transformers get their expressiveness. Each layer can represent increasingly abstract features; a single layer is a severe bottleneck.

### What's Still Missing vs GPT-2

Even the GPT decoder here is a toy. Real GPT-2 small adds:
- 12 layers, 12 attention heads, 768 embedding dimensions (~117M params)
- Pre-layer normalisation (norm before attention, not after)
- Rotary or learned positional embeddings at scale
- Trained on hundreds of millions of tokens, not ~5M
- Weight tying between token embedding and output projection

The purpose of this project is to understand the *architecture*, not match the *scale*.

## Future Improvements

**Model Enhancements**
   - Implement beam search for better text generation
   - Add top-k / nucleus (top-p) sampling
   - Add model checkpointing and early stopping
   - Implement comparison metrics: perplexity, BLEU


**Other**

   - Try LoRA / PEFT Fine-Tuning of GPT-2
   - Explore longer context generation with FlashAttention or Rotary PE
   - Try Instruction-Tuned or RLHF Models, task alignment
   - Quantisation + ONNX/GGUF






## Learning Resources

### Original Papers
- LSTM: [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) (Hochreiter & Schmidhuber, 1997)
- Attention: [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) (Bahdanau et al., 2014)
- BPE: [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) (Sennrich et al., 2015)
- Transformer: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)

### Beginner-Friendly Resources
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual guide to transformers
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) - Great visual explanation of LSTMs
- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) - Comprehensive guide to attention mechanisms
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - Code-focused explanation of transformers

### Video Resources
- [3Blue1Brown's RNN/LSTM video](https://www.youtube.com/watch?v=LHXXI4-IEns) - Visual explanation of RNNs and LSTMs
- [Yannic Kilcher's Attention video](https://www.youtube.com/watch?v=zxQyTK8quyY) - Detailed explanation of attention mechanisms
- [CodeEmporium's Transformer video](https://www.youtube.com/watch?v=TQQlZhbC5ps) - Step-by-step transformer explanation

### Interactive Resources
- [Transformer Playground](https://playground.tensorflow.org/) - Interactive visualization of neural networks
- [Attention Visualization](https://distill.pub/2021/gnn-intro/) - Interactive attention mechanism visualization
- [LSTM Interactive Demo](https://lstm.seas.harvard.edu/) - Interactive LSTM demonstration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 