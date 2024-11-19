# Character-level Language Model (Tiny Shakespeare) - GPT from scratch!

This project is an implementation of a character-level language model, inspired by Andrej Karpathy's work on building neural networks for generating text. The model is based on the Transformer architecture, with a multi-head self-attention mechanism for better sequence understanding. The model is trained on the "Tiny Shakespeare" dataset, which consists of text from the works of William Shakespeare.

### Project Overview

The goal of this project is to create a character-level language model that can generate Shakespeare-style text based on a given prompt. The model uses the Transformer architecture, which is known for its effectiveness in handling sequential data. The core components of the project include:

- **Data Preprocessing**: The raw Shakespeare text is tokenized into individual characters.
- **Model Architecture**: A Transformer-based neural network that learns to predict the next character in a sequence.
- **Training**: The model is trained using the Cross-Entropy Loss function on the character-level predictions.
- **Text Generation**: After training, the model can generate text Shakespeare like text character-by-character.

### Requirements

- Python 3.x
- PyTorch (version 1.10 or later)
- CUDA (optional, for GPU support) - Recommended

To install the dependencies, run:

```bash
pip install torch
```

or install using this [link](https://pytorch.org/get-started/locally/)

### Data

The dataset used in this project is a subset of Shakespeare's works, available from [this URL](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt). To download the data, you can use the following command:

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

### Model Architecture

The model is based on the Transformer architecture with the following components:

- **Embedding Layer**: Converts input characters into embeddings.
- **Multi-Head Self-Attention**: This layer allows the model to focus on different parts of the input sequence to capture long-range dependencies.
- **Feedforward Network**: A simple fully connected network that follows the attention mechanism.
- **Layer Normalization**: Helps stabilize training by normalizing the activations within each layer.
- **Final Output Layer**: A linear layer that maps the model's final hidden states to character logits, which represent the probability distribution over the entire vocabulary.

### Training

The model is trained to predict the next character in a sequence. The training loop runs for a specified number of iterations (`max_iters`), and the loss is evaluated periodically on both the training and validation datasets.

- **Batch Size**: 64
- **Chunk Size**: 256 (Maximum sequence length)
- **Learning Rate**: 3e-4
- **Number of Layers**: 6
- **Embedding Dimensions**: 384
- **Number of Attention Heads**: 6
- **Dropout**: 0.2 for regularization
- **Model size**: ~10M Parameters

During training, the model learns to predict the next character in a sequence, minimizing the [cross-entropy loss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html). 

### Training Procedure

1. Download the data using the provided URL.
2. Split the data into training and validation sets (90% for training, 10% for validation).
3. Train the model on the training data.
4. Periodically evaluate the model on the validation data to monitor progress.
5. Save the generated text after the model is trained.

To start training, simply run the Python script:

```bash
python GPT.py
```

### Text Generation

After training the model, the python file will generate text and save it locally. The model will then predict the next characters in the sequence and generate new text based on the learned patterns.

To generate text:

```python
context = torch.zeros((1,1), dtype=torch.long, device=device)
generated_text = model.generate(context, max_new_tokens=1000)
print(decode(generated_text[0].tolist()))
```

### Output

The model generates text that is similar to the style of Shakespeare's works. After training, the generated text is saved to a file (`output.txt`).

### Example Generated Text

```txt
May your flock near'd to Clarence, back Madam thanks
Dotion into Rome; I'll do found appeal:
```

### Notes

- The model is trained on the Tiny Shakespeare dataset, which is small in comparison to large language models. Thus, the model's ability to generate coherent text might be limited.
- The training process can take some time, especially if you're using a CPU. For faster training, use a GPU.

### Possible Improvements

- **Larger Dataset**: You could train the model on a larger dataset for better performance.
- **Hyperparameter Tuning**: Experimenting with different hyperparameters (e.g., learning rate, batch size, number of layers) could improve the model's ability to generate more coherent text.
- **Pretrained Models**: Starting with a pretrained Transformer model might lead to better results in less time.

### Acknowledgments

- [Andrej Karpathy](https://karpathy.ai/) for the original Char-RNN project and dataset.
- [PyTorch](https://pytorch.org/) for the deep learning framework.
