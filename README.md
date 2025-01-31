# CODE_ai
Outlier AI
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import math

# Define the Transformer-based LLM model
class TransformerLLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, hidden_dim, max_seq_len):
        super(TransformerLLM, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Positional encoding (to give the model information about token positions)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_seq_len)

        # Transformer Encoder layers
        self.encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=num_layers)

        # Output layer (linear transformation to vocab size for prediction)
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        # Apply embedding and positional encoding
        x = self.embedding(x) + self.positional_encoding(x)
        
        # Pass through the transformer encoder
        x = self.transformer_encoder(x)

        # Output layer (linear)
        logits = self.fc_out(x)

        return logits

# Positional encoding to add position information to embeddings
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

# Example dataset class (use your own data loader for large datasets)
class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len=512):
        self.text = text
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        tokenized_text = self.tokenizer(self.text[idx])
        input_ids = torch.tensor(tokenized_text['input_ids'][:self.seq_len])
        return input_ids

# Tokenizer (placeholder for real tokenizer such as from HuggingFace's Transformers)
def simple_tokenizer(text):
    # A simple tokenizer splitting by spaces (replace this with a real tokenizer for practical use)
    return {'input_ids': [ord(c) for c in text]}

# Hyperparameters
vocab_size = 5000  # Example vocab size
embedding_dim = 256
num_heads = 8
num_layers = 4
hidden_dim = 1024
max_seq_len = 512
batch_size = 8
learning_rate = 0.001
epochs = 10

# Create the model
model = TransformerLLM(vocab_size, embedding_dim, num_heads, num_layers, hidden_dim, max_seq_len)

# Optimizer and Loss Function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

# Example data (text data in a list of strings)
sample_text = ["Hello world.", "This is a simple example of a transformer model.", "You can modify the dataset."]
dataset = TextDataset(sample_text, simple_tokenizer)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for batch in data_loader:
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch)  # Shape: [batch_size, seq_len, vocab_size]
        
        # Prepare targets (next token prediction)
        targets = batch[:, 1:].contiguous()  # Shift by 1 position to predict next token

        # Compute loss
        loss = loss_fn(outputs.view(-1, vocab_size), targets.view(-1))
        
        # Backpropagation
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(data_loader)}")
