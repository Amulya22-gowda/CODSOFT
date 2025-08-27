import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
import matplotlib.pyplot as plt

# --------------------------
# 1. Load IAM dataset
# --------------------------
dataset = load_dataset("Teklia/IAM-line")

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 512)),   # normalize all images
    transforms.ToTensor()
])

# --------------------------
# 2. Vocabulary builder
# --------------------------
class CharVocab:
    def __init__(self, texts):
        chars = sorted(list(set("".join(texts))))
        self.char2idx = {c: i+1 for i, c in enumerate(chars)}  # start from 1
        self.char2idx["<PAD>"] = 0
        self.idx2char = {i: c for c, i in self.char2idx.items()}

    def encode(self, text):
        return [self.char2idx.get(c, 0) for c in text]

    def decode(self, seq):
        return "".join([self.idx2char.get(i, "") for i in seq if i != 0])

# --------------------------
# 2. Vocabulary builder
# --------------------------
class CharVocab:
    def __init__(self, texts):
        chars = sorted(list(set("".join(texts))))
        self.char2idx = {c: i+1 for i, c in enumerate(chars)}  # start from 1
        self.char2idx["<PAD>"] = 0
        self.idx2char = {i: c for c, i in self.char2idx.items()}

    def encode(self, text):
        return [self.char2idx.get(c, 0) for c in text]

    def decode(self, seq):
        return "".join([self.idx2char.get(i, "") for i in seq if i != 0])

# --------------------------
# Helper: CTC Greedy Decoder
# --------------------------
def ctc_decode(pred_seq):
    prev = -1
    decoded = []
    for p in pred_seq:
        if p != prev and p != 0:   # skip blanks & repeats
            decoded.append(p)
        prev = p
    return vocab.decode(decoded)

# --------------------------
# 3. Custom Dataset
# --------------------------
class IAMDataset(Dataset):
    def __init__(self, split, transform, vocab):
        self.data = dataset[split]
        self.transform = transform
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img = self.transform(sample["image"])
        text = torch.tensor(self.vocab.encode(sample["text"]), dtype=torch.long)
        return img, text

# Collate function (handles variable-length text sequences)
def collate_fn(batch):
    imgs, texts = zip(*batch)
    imgs = torch.stack(imgs)
    return imgs, texts

# --------------------------
# 4. Build vocab + loaders
# --------------------------
all_texts = [s["text"] for s in dataset["train"]]
vocab = CharVocab(all_texts)

train_dataset = IAMDataset("train", transform, vocab)
val_dataset   = IAMDataset("validation", transform, vocab)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# --------------------------
# 5. Model
# --------------------------
class HandwritingRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size):
        super(HandwritingRNN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)  # bidirectional

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

# --------------------------
# 6. Training setup
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HandwritingRNN(input_dim=512, hidden_dim=256, vocab_size=len(vocab.char2idx)).to(device)

criterion = nn.CTCLoss(blank=0)   # use <PAD>=0 as blank
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --------------------------
# 7. Training loop
for epoch in range(2): 
    model.train()
    total_loss = 0

    for imgs, texts in train_loader:
        imgs = imgs.to(device)
        targets = torch.cat(texts).to(device)

        B, C, H, W = imgs.shape
        imgs_seq = imgs.view(B, H, -1)   # (B, seq_len, input_dim)

        outputs = model(imgs_seq).log_softmax(2)

        input_lengths = torch.full(size=(B,), fill_value=outputs.size(1), dtype=torch.long)
        target_lengths = torch.tensor([t.size(0) for t in texts], dtype=torch.long)

        loss = criterion(outputs.permute(1, 0, 2), targets, input_lengths, target_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
# Quick sample check
model.eval()
with torch.no_grad():
    imgs, texts = next(iter(val_loader))
    imgs = imgs.to(device)
    B, C, H, W = imgs.shape
    imgs_seq = imgs.view(B, H, -1)
    outputs = model(imgs_seq).log_softmax(2)
    pred = outputs.argmax(-1)[0].cpu().numpy()
    pred_text = ctc_decode(pred)
    true_text = vocab.decode(texts[0].numpy())
    print(f"Sample True: {true_text}")
    print(f"Sample Pred: {pred_text}")

    # save checkpoint
    torch.save(model.state_dict(), f"handwriting_rnn_epoch{epoch+1}.pth")


print("✅ Model saved as handwriting_rnn.pth")

# --------------------------
# 8. Evaluation
# --------------------------
model.eval()
sample = dataset["validation"][0]
img = transform(sample["image"]).unsqueeze(0).to(device)
true_text = sample["text"]

with torch.no_grad():
    B, C, H, W = img.shape
    img_seq = img.view(B, H, -1)
    outputs = model(img_seq).log_softmax(2)
    pred = outputs.argmax(-1)[0].cpu().numpy()
    pred_text = ctc_decode(pred)


print("True :", true_text)
print("Pred :", pred_text)

plt.imshow(sample["image"], cmap="gray")
plt.axis("off")
plt.title(f"True: {true_text}\nPred: {pred_text}")
plt.show()

# --------------------------
# Validation Accuracy
# --------------------------
correct, total = 0, 0
for imgs, texts in val_loader:
    imgs = imgs.to(device)
    with torch.no_grad():
        B, C, H, W = imgs.shape
        imgs_seq = imgs.view(B, H, -1)
        outputs = model(imgs_seq).log_softmax(2)
        pred = outputs.argmax(-1)[0].cpu().numpy()
        pred_text = ctc_decode(pred)
        true_text = vocab.decode(texts[0].numpy())

        if pred_text == true_text:
            correct += 1
        total += 1

print(f"Validation Accuracy: {correct/total:.2%}")
