#imports

import time
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
batch_size = 64 #independent chunks that we want to process in parallel
chunk_size = 256 #maximum context length
max_iters= 5000 #training loop
eval_intervals = max_iters // 10 #printing results
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else "cpu"
eval_iters = 200 
n_embd = 384 #number of embedding dimensions 384/6 = 64 dimensional head
n_head = 6 #heads of self attention
n_layer = 6
dropout = 0.2 #regularization - 20% neurons disabled
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

class Head(nn.Module):
    "One head of the self-attention"
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.values = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(chunk_size, chunk_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        #compute the attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 #B, T, C @ B, C, T ---> B, T, T
        wei = wei.masked_fill(self.tril[:T, :T]==0, float("-inf")) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        #perfoming the weigthed aggregation of the values
        v = self.values(x) # B, T, C
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
    
class MultiHeadAttention(nn.Module):
    """Multiple heads of self attention run in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)

        return out
        
class FeedForward(nn.Module):
    """A simple linear layer followed by non-linearity"""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
        

#Simple bigrammodel
class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(chunk_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.layernorm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        #idx and targets are (B, T) tenor of integers
        #logits are scores for the next character in the sequence
        token_embd = self.token_embedding_table(idx) #(B, T , C) Batch = 8, Time = 10, Channels = Vocab size
        position_embd = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = token_embd + position_embd
        x = self.blocks(x) # (B, T, C)
        x = self.layernorm(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

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

            idx_cond = idx[:, -chunk_size:]

            logits, loss = self(idx_cond) #predictions
            logits = logits[:,-1, :] #Want the lass occurance to pluck out T, now becomes (B, C)
            probs = F.softmax(logits, dim=-1) #(B, C)

            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx

            
    
model = BigramLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

#creating the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)


start_time = time.time()
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

print("--- %s Training time in Minutes ---" % ((time.time() - start_time)/60))

#generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
#print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

open('output.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))



