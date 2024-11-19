#imports

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(42) #replicability

#getting the data
#wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
#inspecting the input text
with open("input.txt", "r") as f:
    text = f.read()

#hyperparameters
batch_size = 32
chunk_size = 10
max_iters= 3000
eval_intervals = max_iters // 10
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else "cpu"
eval_iters = 200
# --------------


#let's check the unique characters in the text file - this is the vocabulary we are dealing with
vocab = sorted(list(set(text)))
vocab_size = len(vocab)

print((" ".join(vocab)))
print("vocab size = ", vocab_size)


# ### Encoder and Decoder
encoder_look_up = {char : i for i, char in enumerate(vocab)}
decoder_look_up = {i : char for i, char in enumerate(vocab)}

def encode(string):
        return [encoder_look_up[char] for char in string]

def decode(coded_list):
        return "".join(decoder_look_up[code] for code in coded_list)


# Encoding the input and store it in a torch tensor
data = torch.tensor(encode(text), dtype=torch.long) #64-bit integer (signed) - We can also use the unsigned 64 bit integer but the torch documentation says limited support for it, so I am going to use the signed one for sanity



### Train Val Split
n = int(0.9 * len(data))
train_data = data[:n] #no shuffling as the sequence is important
val_data = data[n:]


# ### Batching
def get_batches(split):
    """
    Get batch of data to either "train" or "val"
    """
    
    data = train_data if split.lower()=="train" else val_data
    positions = torch.randint(len(data)-chunk_size, (batch_size,))
    x = torch.stack([data[i:i+chunk_size] for i in positions])
    y = torch.stack([data[i+1:i+chunk_size+1] for i in positions])
    x , y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batches(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


#Simple bigrammodel
class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        #idx and targets are (B, T) tenor of integers
        #logits are scores for the next character in the sequence
        logits = self.token_embedding_table(idx) #(B, T , C) Batch = 8, Time = 10, Channels = Vocab size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view( B * T, C) # This is done because pytorch expects the channels to be the second dimention
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        #idx is the (B, T) array of index in the current context

        for _ in range(max_new_tokens):
            logits, loss = self(idx) #predictions
            logits = logits[:,-1, :] #Want the lass occurance to pluck out T, now becomes (B, C)
            probs = F.softmax(logits, dim=-1) #(B, C)

            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx

            
    
model = BigramLanguageModel(vocab_size)
m = model.to(device)


#creating the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)


for iter in range(max_iters):

    if iter%eval_intervals==0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb, yb = get_batches("train")

    #evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))





