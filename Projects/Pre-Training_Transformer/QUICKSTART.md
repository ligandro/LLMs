# Transformer Pretraining Project - Quick Reference

## Project Structure
```
Transformer_Pretraining/
├── README.md                      # Project documentation
├── pretrain_transformer.ipynb     # Main training notebook
├── config.yaml                    # Configuration file
├── data/
│   └── corpus.txt                 # Training text corpus
├── models/
│   └── checkpoints/               # Model checkpoints (created during training)
└── modules/
    ├── __init__.py                # Package initializer
    ├── transformer.py             # Transformer architecture
    ├── trainer.py                 # Training utilities
    └── data_utils.py              # Data loading and preprocessing
```

## Quick Start

1. **Open the notebook**:
   ```
   pretrain_transformer.ipynb
   ```

2. **Run all cells** to:
   - Load training data
   - Build vocabulary
   - Initialize transformer model
   - Train the model
   - Generate text

3. **Or use modules directly**:
   ```python
   from modules import TransformerLM, SimpleTokenizer, train_epoch
   ```

## Files Overview

### Core Modules
- **`transformer.py`**: Complete transformer implementation
  - PositionalEncoding
  - MultiHeadAttention
  - TransformerBlock
  - TransformerLM

- **`data_utils.py`**: Data preprocessing
  - SimpleTokenizer (word-level)
  - TextDataset (for PyTorch)

- **`trainer.py`**: Training utilities
  - train_epoch()
  - evaluate()
  - generate_text()
  - get_lr_scheduler()

### Data
- **`corpus.txt`**: Sample training corpus (NLP concepts text)
- Replace with your own text data for better results

### Configuration
- **`config.yaml`**: Hyperparameters
  - Model architecture settings
  - Training parameters
  - Data configuration

## Training

The notebook handles everything, but you can also train programmatically:

```python
import torch
from modules import TransformerLM, SimpleTokenizer, TextDataset, train_epoch

# Load data
with open('data/corpus.txt') as f:
    text = f.read()

# Build tokenizer
tokenizer = SimpleTokenizer(vocab_size=5000)
tokenizer.build_vocab(text)

# Create dataset
dataset = TextDataset(text, tokenizer, seq_length=128)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
model = TransformerLM(vocab_size=len(tokenizer.word2idx))

# Train
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()
loss = train_epoch(model, loader, criterion, optimizer, scheduler, device='cpu')
```

## Model Size

Default configuration: **~8M parameters**
- Adjustable via config.yaml
- Trade-off between performance and training time

## Requirements

```
torch>=2.0.0
numpy
matplotlib
tqdm
pyyaml (optional, for config loading)
```

## Tips

1. **For faster training**: Use GPU by setting device='cuda'
2. **Better results**: Train on larger corpus (Wikipedia, books, etc.)
3. **Monitor training**: Check loss curves and sample generations
4. **Experiment**: Try different hyperparameters in config.yaml

Enjoy training your transformer! 🚀
