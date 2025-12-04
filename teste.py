import torch
import torch.nn as nn

# Configurações
head_size = 4
num_heads = 3
n_embd = head_size * num_heads # 12

# Simulação da classe Head (simplificada)
class Head(nn.Module):
    def __init__(self, size):
        super().__init__()
        # Cada cabeça projeta para um tamanho pequeno
        self.linear = nn.Linear(n_embd, size) ## -- >> 12, 
    
    def forward(self, x):
        return self.linear(x)

# Nossa classe MultiHead (Explodida para teste)
class MultiHeadSim(nn.Module):
    def __init__(self):
        super().__init__()
        # Cria 3 cabeças independentes
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # Projeção final para misturar
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x): # Forward escrito corretamente!
        # 1. Cada cabeça processa
        head_outputs = [h(x) for h in self.heads]
        print(head_outputs)
        
        # 2. Concatenamos (Cole os vetores lado a lado)
        out = torch.cat(head_outputs, dim=-1)
        print(out)
        print(f"Shape após CAT: {out.shape}") # Deve ser (Batch, T, 12)
        
        # 3. Projeção
        out = self.proj(out)
        return out

# Teste
model = MultiHeadSim()
x = torch.randn(1, 10, n_embd) # (Batch, Time, Channels=12)
output = model(x)