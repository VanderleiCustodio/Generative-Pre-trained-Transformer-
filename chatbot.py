
import torch
import torch.nn as nn
from torch.nn import functional as F
import pickle
# HyperParameters


# b = batch_size
# t = block_size
# c = vocab_Size


device = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)

batch_size = 128
block_size = 64
max_iters = 100
learning_rate = 3e-3
eval_iters = 100
n_embd = 128
n_head = 8
n_layer = 1
dropout = 0.2

print(device)


torch.manual_seed(1337)
chars = ''
with open("lolbas.txt",'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(set(text))
    
vocab_Size = len(chars)


stoi = {ch:i for i,ch in enumerate(chars)}
Itos = {i:ch for i,ch in enumerate(chars)}
    
encode = lambda x: [stoi[c] for c in x]
decode = lambda x: "".join([Itos[c] for c in x])
    
data = torch.tensor(encode(text), dtype=torch.long)


class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd,head_size, bias=False)
        self.value = nn.Linear(n_embd,head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) ## elimina overhead 
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B,T, C = x.shape
        
        k = self.key(x)
        q = self.query(x)
        
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x)
        
        out = wei @ v 
        
        return out 
class MultiHeadAttention(nn.Module):
    """ multiple heads of self attention in parallel  """
    
    def __init__(self, n_embd, head_size):
        super().__init__()
        
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj  = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) ### (B, T, F) -> B, T, [h1,h1,h1,h2,h2,h2,h3,h3,h3...]
        out = self.dropout(self.proj(out))
        return out
    
class FeedFoward(nn.Module):
    """  a simple linear layer followed by a non-linearity """
    
    def __init__(self, n_embd):
        super().__init__()
        
        self.net = nn.Sequential(
            
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
            return self.net(x)

class Block(nn.Module):
    def __init__(self,n_embd, n_head):
        
        super().__init__()
        
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd) 
        
    def forward(self, x):
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
        
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # ex --> [-400, 300] --> [1.44, 1.2300], aqui eu to falando que vai ser 32 de dimensão [1.. 2.. ..3.]
        self.lm_head = nn.Linear(n_embd, vocab_Size) #"""É um perceptron linear tem pesos W de tamanho 96 × 32 Tem bias b de tamanho 96 Transforma 32 features → 96 features É apenas: y = x @ W^T + b """
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
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
    
     
    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            index_cond = index[:, -block_size:]
            # get the predictions
            logits, loss = self.forward(index_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1) # (B, T+1)
        return index
    
    
model = GPTLanguageModel(vocab_Size)
m = model.to(device)

with open('mode-01.pk','wb') as f:
    pickle.dump(model, f)

while True:
    prompt = input('Prompt: \n')
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    generated_chars = decode(m.generate(context.unsqueeze(0), max_new_tokens=150)[0].tolist())
    print(f'Completion: \n {generated_chars}')
