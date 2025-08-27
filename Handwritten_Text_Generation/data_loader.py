import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset

# Load IAM dataset
dataset = load_dataset("Teklia/IAM-line")

# Transform for images
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 512)),   # fix height=128, width=512
    transforms.ToTensor()            # convert to tensor, normalize 0–1
])

# Character encoding
class CharVocab:
    def __init__(self, texts):
        chars = sorted(list(set("".join(texts))))
        self.char2idx = {c: i+1 for i, c in enumerate(chars)}  # start from 1
        self.char2idx["<PAD>"] = 0
        self.idx2char = {i: c for c, i in self.char2idx.items()}

    def encode(self, text):
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def decode(self, seq):
        return "".join([self.idx2char[i] for i in seq if i != 0])

# Custom Dataset
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

# 🔹 Define collate_fn BEFORE using it
def collate_fn(batch):
    imgs, texts = zip(*batch)
    imgs = torch.stack(imgs, 0)

    fixed_len = 128  # must match image height
    padded_texts = torch.zeros(len(texts), fixed_len, dtype=torch.long)

    for i, t in enumerate(texts):
        length = min(t.size(0), fixed_len)  # trim if too long
        padded_texts[i, :length] = t[:length]

    return imgs, padded_texts



# Build vocab from training data
all_texts = [s["text"] for s in dataset["train"]]
vocab = CharVocab(all_texts)

# Create train loader (now collate_fn is defined!)
train_dataset = IAMDataset("train", transform, vocab)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
