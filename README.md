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

### **Character-Level Attention** (`char-level-attention.py`)
   - Attention-based model operating at the character level
   - Combines the benefits of attention with character-level modeling
   - More flexible but computationally expensive
   - Third in expected performance
   - Good intermediate step between LSTM and transformer models
   - ~56k params

### **Word-Level Attention** (`word-level-attention.py`)
   - Implements a transformer-based architecture
   - Uses multi-head self-attention mechanism
   - Includes positional encoding and transformer blocks
   - More sophisticated approach for word-level generation
   - ~5.7M params

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


```bash
$ python subword-attention.py 

Epoch 5/5
20768/20768 [==============================] - 165s 8ms/step - loss: 4.8897
to be, or not to be to your lordship upon a man? a widow. the brother the shallow lords to music, the thieves in patience, i are sick, calling forlorn will as can approach this man. , this, but so most earth is over my beautyuous
```



## Future Improvements

**Model Enhancements**
   - Add Top-k Sampling or Temperature Scaling
   - Implement beam search for better text generation
   - Experiment with larger transformer architectures
   - Add model checkpointing and early stopping
   - Implement model comparison metrics, perplexity, BLEU


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