import torch
import torch.nn as nn
import torch.optim as optim

# 1. Tiny toy dataset for sentiment
texts = [
    # strong positive (label 1)
    "i love this",
    "this is great",
    "amazing work",
    "i really like this",
    "this is fantastic",
    "this makes me happy",
    "i enjoy this a lot",
    "i'm very satisfied",
    "this worked perfectly",
    "i'm impressed by this",

    # mild positive (label 1)
    "this is okay",
    "it's not bad",
    "pretty good overall",

    # strong negative (label 0)
    "i hate this",
    "this is bad",
    "terrible experience",
    "i really dislike this",
    "this is awful",
    "this makes me sad",
    "i regret this",
    "i'm very disappointed",
    "this is the worst",

    # mild negative (label 0)
    "this could be better",
    "not great",
    "i'm not very happy with this",

    # NEUTRAL EXAMPLES (we’ll treat as mid, but label them 0.5-ish via thresholds later)
    "this is fine",
    "it's okay",
    "i don't mind this",
    "this is acceptable",
    "i have no strong feelings about this",
    "this is average",
    "this is neither good nor bad",
    "this is just okay"
]

labels = [
    # positives (1)
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1,
    # negatives (0)
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0,
    # neutrals (0.5)
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    0.5, 0.5
]

# 2. Build a simple vocabulary from the dataset
word2idx = {}
for text in texts:
    for word in text.split():
        if word not in word2idx:
            word2idx[word] = len(word2idx)

vocab_size = len(word2idx)

def text_to_vector(text: str) -> torch.Tensor:
    vec = torch.zeros(vocab_size)
    for word in text.lower().split():
        if word in word2idx:
            vec[word2idx[word]] += 1
    return vec

# Turn all texts into vectors
X = torch.stack([text_to_vector(t) for t in texts])  # shape: [N, vocab_size]
y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # shape: [N, 1]

# 3. Define a tiny neural net model
class TinySentimentModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)
        return torch.sigmoid(logits)  # output between 0 and 1


model = TinySentimentModel(input_dim=vocab_size, hidden_dim=16)

# 4. Loss + optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# 5. Training loop
for epoch in range(200):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/200, loss={loss.item():.4f}")

# 6. Save the trained model + vocab
checkpoint = {
    "model_state_dict": model.state_dict(),
    "word2idx": word2idx,
}
torch.save(checkpoint, "nuerova_sentiment.pt")
print("Saved tiny NuerovaAI sentiment model to nuerova_sentiment.pt")