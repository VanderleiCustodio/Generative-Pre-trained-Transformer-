
import torch
import torch.nn as nn
from torch.nn import functional as F

# HyperParameters

batch_size = 32 ## how mane independentes sequences will we process in paralel ?
block_size = 8 # what is the maximum context lenght for predictions ?
max_iters = 10000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
n_head = 4
n_layer = 4
dropout = 0.2 # 20% dos neurionios vao cair para zero


#____________

torch.manual_seed(1337)

with open("lolbas.txt",'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(set(text))
    vocab_Size = len(chars)

    stoi = {ch:i for i,ch in enumerate(chars)}
    Itos = {i:ch for i,ch in enumerate(chars)}
    
    encode = lambda x: [stoi[c] for c in x]
    decode = lambda x: "".join([Itos[c] for c in x])
    
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n] ## first 90% of the data set is up to train, rest val
val_data = data[n:]

def get_batch(split):
    
    # generate a small number of data inputs x and targets Y
    data = train_data if split == 'train' else val_data
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    x, y = x.to(device), y.to(device)
    
    return x, y 

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
        
    model.train()
    
    return out


# b = batch_size
# t = block_size
# c = vocab_Size


class MultiHeadAttention(nn.Module):
    """ multiple heads of self attention in parallel  """
    
    def __init__(self):
        super().__init__()
        
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def foward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) ### (B, T, F) -> B, T, [h1,h1,h1,h2,h2,h2,h3,h3,h3...]
        out = self.dropout(self.proj(out))
        return out
    
class FeedFoward(nn.Module):
    """  a simple linear layer followed by a non-linearity """
    
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
        
        def forward(self, x):
            return self.net(x)

class Block:
    def __init__(self):
        
        super().__init__()
        
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd) 
        
    def foward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x +y)
        
        return x    

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_Size):
        super().__init__()
        
        self.token_embedding_table = nn.Embedding(vocab_Size, n_embd) 
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)
        
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # ex --> [-400, 300] --> [1.44, 1.2300], aqui eu to falando que vai ser 32 de dimensão [1.. 2.. ..3.]
        self.lm_head = nn.Linear(n_embd, vocab_Size) #"""É um perceptron linear tem pesos W de tamanho 96 × 32 Tem bias b de tamanho 96 Transforma 32 features → 96 features É apenas: y = x @ W^T + b """
    
    def forward(self, idx, targets=None):
        
        tok_emb = self.token_embedding_table(idx) # [2, 3, 4, 5,  6,]
        pos_emb = self.positional_embedding_table(torch.arange(T, device=device))
        
        
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        
        logits = self.lm_head(x)    
             
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            
            logits = logits.view(B*T,C) # batch + block
            targets = targets.view(B*T)
            
            
            loss = F.cross_entropy(logits,targets)
        
        return logits, loss
    
    def generate(self, idx, number_of_maxtokens):
        # idx is (B,T) array of indices in the current position 
        
        for _ in range(number_of_maxtokens):
            
            #get the predictions 
            logits, loss = self(idx)
            
            #focus only on the last time step
            logits = logits[:,-1,:] # becomes (B,C)
            
            # apply softmax to get probabilities
            
            probs = F.softmax(logits, dim=1) # B, C
            
            # sample from the distribuition
            idx_next = torch.multinomial(probs, num_samples=1) # b,1
            
            idx = torch.cat((idx, idx_next), dim=1) # (B, T +1)
            
        return idx
    
    
model = GPTLanguageModel(vocab_Size)
m = model.to(device)


### crete pytorch optimizer 

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

## take the gradients and update the paramets using the gradiantes 


for iter in range(eval_iters):
    
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4}')
        
    ## sample batch size 
    xb, yb = get_batch('train')
    
    # evaluate the loss
    
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)    
    loss.backward()
    
    optimizer.step()
    
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, number_of_maxtokens=400)[0].tolist()))
