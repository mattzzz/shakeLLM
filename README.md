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

## Project Overview

The project contains several implementations of text generation models, ordered from simplest to most complex:

1. **Character-Level LSTM** (`char-level-lstm.py`)
   - Simple LSTM-based model operating at the character level
   - Uses embedding layer followed by LSTM and dense layers
   - Good for understanding basic sequence modeling
   - Baseline implementation with simplest architecture
   - Expected to perform fourth due to basic architecture

2. **Character-Level Attention** (`char-level-attention.py`)
   - Attention-based model operating at the character level
   - Combines the benefits of attention with character-level modeling
   - More flexible but computationally expensive
   - Third in expected performance
   - Good intermediate step between LSTM and transformer models

3. **Subword Attention** (`subword-attention.py`)
   - Similar to word-level attention but operates on subword units
   - Uses BPE (Byte Pair Encoding) for tokenization
   - Good balance between character and word-level modeling
   - Second best performance expected due to efficient tokenization
   - Modern approach to handling vocabulary

4. **Word-Level Attention** (`word-level-attention.py`)
   - Implements a transformer-based architecture
   - Uses multi-head self-attention mechanism
   - Includes positional encoding and transformer blocks
   - More sophisticated approach for word-level generation
   - Expected to perform best due to modern architecture and word-level understanding

## Learning Resources

### Original Papers
- LSTM: [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) (Hochreiter & Schmidhuber, 1997)
- Attention: [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) (Bahdanau et al., 2014)
- Transformer: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- BPE: [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) (Sennrich et al., 2015)

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

## Model Performance Comparison

### Expected Performance Ranking

1. **Word-Level Attention**
   - Best performance due to transformer architecture
   - Superior at capturing long-range dependencies
   - Better semantic understanding through word-level modeling
   - Modern architecture with attention mechanisms
   - Includes best practices (layer normalization, dropout)

2. **Subword Attention**
   - Strong performance with efficient tokenization
   - Good balance of flexibility and efficiency
   - Can handle out-of-vocabulary words
   - Maintains benefits of attention mechanism
   - Slightly lower performance than word-level due to longer sequences

3. **Character-Level Attention**
   - Moderate performance with high flexibility
   - Can generate any character sequence
   - More computationally expensive
   - Struggles with maintaining word-level semantics
   - Requires longer training to learn patterns

4. **Character-Level LSTM**
   - Basic performance with simple architecture
   - Limited in capturing long-range dependencies
   - No attention mechanism
   - More prone to generating nonsensical text
   - Good baseline for comparison

### Performance Factors

The ranking is based on several key factors:
1. **Architecture sophistication**: Transformer-based models generally outperform LSTM
2. **Tokenization level**: Word-level > Subword > Character-level for most tasks
3. **Model capacity**: More complex architectures can learn more sophisticated patterns
4. **Training efficiency**: Models that better handle long sequences and dependencies
5. **Generation quality**: Ability to maintain coherent and meaningful text

## Dataset

The project uses Shakespeare's complete works as the training data (`shakespeare.txt`). The text is preprocessed and tokenized according to the specific model requirements.

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy

## Usage

First download the complete works of Shakespeare from Project Gutenberg to shakespeare.txt
Each model can be run independently. For example, to run the character-level LSTM model:

```bash
python char-level-lstm.py
```

The models will train on the Shakespeare dataset and can generate new text given a seed phrase.

## Model Architecture Details

### Word-Level Attention
- Embedding layer (64 dimensions)
- Positional encoding
- Transformer block with:
  - Multi-head self-attention (2 heads)
  - Feed-forward network
  - Layer normalization
  - Dropout (0.1)
- Sequence length: 10 words

### Subword Attention
- Similar architecture to word-level attention
- BPE tokenization
- Optimized for handling subword units

### Character-Level Attention
- Attention-based architecture
- Character-level tokenization
- Longer sequence lengths

### Character-Level LSTM
- Embedding layer (64 dimensions)
- LSTM layer (128 units)
- Dense output layer with softmax activation
- Sequence length: 100 characters

## Future Improvements

1. **Model Enhancements**
   - Implement beam search for better text generation
   - Add temperature parameter for controlling randomness
   - Experiment with larger transformer architectures
   - Add model checkpointing and early stopping
   - Implement model comparison metrics

2. **Training Improvements**
   - Implement learning rate scheduling
   - Add validation split and monitoring
   - Implement gradient clipping
   - Add model evaluation metrics
   - Add performance benchmarking

3. **Feature Additions**
   - Add command-line interface for text generation
   - Implement model saving and loading
   - Add support for different datasets
   - Create a web interface for text generation
   - Add model comparison visualization

4. **Code Quality**
   - Add proper logging
   - Implement unit tests
   - Add type hints
   - Create proper configuration management
   - Add performance profiling

5. **Documentation**
   - Add detailed API documentation
   - Create usage examples
   - Add model architecture diagrams
   - Include performance benchmarks
   - Add model comparison results

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 