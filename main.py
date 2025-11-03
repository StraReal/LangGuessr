from aksharamukha import transliterate
from datasets import load_dataset
from collections import defaultdict
import random
from itertools import chain
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from unidecode import unidecode
import csv
import regex

def is_latin(word):
    letters = regex.findall(r'\p{L}', word)
    return all(regex.match(r'\p{Latin}', ch) for ch in letters)

def load_index_csv(path="indexing.csv"):
    blocks = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lang = row["lang"]
            blocks[lang] = (int(row["start"]), int(row["end"])-1)
    return blocks
def smart_transliterate(text):
    # Only transliterate if the text is not Latin
    if not is_latin(text):
        return unidecode(transliterate.process('autodetect', 'latn', text, param="script_code"))
    else:
        return unidecode(text)
def cprint(text, color="w"):
    """
    :param text: text to print
    :param color: color to print in (red, green, yellow, blue, magenta, cyan, white)
    :return: None
    """
    colors = {
        "r": "\033[91m",
        "g": "\033[92m",
        "y": "\033[93m",
        "b": "\033[94m",
        "m": "\033[95m",
        "c": "\033[96m",
        "w": "\033[97m",
    }
    RESET = "\033[0m"
    print(f"{colors.get(color, colors['w'])}{text}{RESET}")
def encode_to_tensor(inpword):
    word_idx = [char2idx.get(c, 0) for c in inpword[:max_len]]
    word_idx += [0] * (max_len - len(word_idx))
    return torch.tensor([word_idx], dtype=torch.long)
def encode_word(word):
    # Convert each character to index, truncate if needed
    word_idx = [char2idx.get(c, 0) for c in word[:max_len]]
    # Pad with zeros if shorter
    word_idx += [0] * (max_len - len(word_idx))
    return word_idx

def clean_word(inpword: str) -> str:
    """Lowercase, transliterate, and remove non-alpha characters."""
    inpword = smart_transliterate(inpword.lower())
    return ''.join(c for c in inpword if c.isalpha())



def choose_model(folder="models"):
    os.makedirs(folder, exist_ok=True)
    files = sorted([f for f in os.listdir(folder) if f.endswith(".pth")])

    cprint("Options:", "c")
    cprint("0) Train new model", "g")
    for i, f in enumerate(files, 1):
        cprint(f"{i}) {f}", "y")

    while True:
        try:
            choice = int(input("Enter your choice: "))
            if 0 <= choice <= len(files):
                break
            cprint(f"Please enter a number between 0 and {len(files)}", "r")
        except ValueError:
            cprint("Invalid input, enter a number.", "r")

    return (True, None) if choice == 0 else (False, os.path.join(folder, files[choice - 1]))

train, model_path = choose_model()

# === Choose which languages to include ===
selected_langs = [
    "eng",
    "fra",
    "spa",
    "hin",
    "ind",
    "por",
    "ben",
    "arb",
    "rus",
    "cmn",
]

y_test, X_test = None, None

if train:
    # === 1. Load the PanLex dataset ===
    dataset = load_dataset("lbourdois/panlex")["train"]
    index_blocks = load_index_csv("indexing.csv")
    # === 2. Initialize data containers ===
    previous_lang = None
    disc_langs = 0
    count=0
    max_per_lang = 10000  # how many samples to collect per language
    lang_words = defaultdict(list)
    lang_counts = defaultdict(int)
    # === 3. Stream through the dataset ===
    for lang in selected_langs:
        start, end = index_blocks[lang]
        print(f"Processing {lang} [{start}:{end}] ({end - start} entries)")
        subset = dataset.select(range(start, end))
        n = len(subset)
        while lang_counts[lang] < 10000:
            idx = random.randint(0, n - 1)  # random index in subset
            entry = subset[idx]
            word = (entry['vocab'] or "").strip()

            if len(word) < 3 or not word.islower():
                continue

            word_clean = clean_word(word)
            if not word_clean or len(word_clean) < 3:
                continue

            lang_words[lang].append(word_clean)
            lang_counts[lang] += 1

    for lang in lang_words:
        if len(lang_words[lang]) > max_per_lang:
            lang_words[lang] = random.sample(lang_words[lang], max_per_lang)

    # === 4. Print summary ===
    print("\nCollected samples per language:")
    for lang in selected_langs:
        print(f"{lang}: {len(lang_words[lang])}")

    # === 5. Flatten everything into a single list for training ===
    all_samples = [
        {"vocab": word, "lang": lang}
        for lang, words in lang_words.items()
        for word in words
    ]
    # Shuffle to mix languages
    random.shuffle(all_samples)
    total_samples = len(all_samples)
    print(f"\nTotal collected samples: {total_samples}")
    entries = []
    for i in range(20):
        entries.append(random.randint(0,total_samples))
    entries.sort()
    # === 7. Separate inputs and labels ===
    words = [entry["vocab"] for entry in all_samples]
    langs = [entry["lang"] for entry in all_samples]
    for ent in entries:
        print(f"Example: {words[ent]} -> {langs[ent]}")

    # Collect all unique characters
    all_chars = sorted(set(chain.from_iterable(words)))
    char2idx = {c: i+1 for i, c in enumerate(all_chars)}  # +1 to reserve 0 for padding
    idx2char = {i: c for c, i in char2idx.items()}
    vocab_size = len(char2idx) + 1  # +1 for padding
    print(f"Vocabulary size: {vocab_size}")
    max_len = 40

    X = [encode_word(word) for word in words]

    lang2idx = {lang: i for i, lang in enumerate(selected_langs)}
    y = [lang2idx[lang] for lang in langs]

    X = torch.tensor(X, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

class CharCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # [batch, seq_len, embed_dim]
        x = x.permute(0, 2, 1)  # [batch, embed_dim, seq_len] for Conv1d
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)  # global max pooling
        x = self.fc(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"

if train:
    model = CharCNN(vocab_size=vocab_size, embed_dim=32, num_classes=len(selected_langs), max_len=max_len).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    batch_size = 64
    epochs = 10

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")
    model_path = f"models/{input('Insert model name: ')}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'char2idx': char2idx,
        'idx2char': idx2char,
        'lang2idx': lang2idx,
        'idx2lang': {i: l for l, i in lang2idx.items()},
        'max_len': max_len
    }, model_path)

    print(f"✅ Model saved to {model_path}")
else:
    checkpoint = torch.load(model_path, map_location=device)
    vocab_size = len(checkpoint['char2idx']) + 1
    max_len = checkpoint['max_len']
    selected_langs = list(checkpoint['lang2idx'].keys())

    model = CharCNN(
        vocab_size=vocab_size,
        embed_dim=32,
        num_classes=len(selected_langs),
        max_len=max_len
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Also restore char2idx, idx2char, lang2idx, idx2lang if needed for encoding/inference
    char2idx = checkpoint['char2idx']
    idx2char = checkpoint['idx2char']
    lang2idx = checkpoint['lang2idx']
    idx2lang = checkpoint['idx2lang']

model.eval()

with torch.no_grad():
    if None not in (X_test, y_test):
        X_test, y_test = X_test.to(device), y_test.to(device)
        preds = model(X_test)
        acc = (preds.argmax(1) == y_test).float().mean()
        print(f"Test accuracy: {acc:.4f}")

def predict_top2_prob(inpword, model, idx2lang):
    model.eval()
    with torch.no_grad():
        x = inpword
        x = x.to(next(model.parameters()).device)
        logits = model(x).squeeze(0)
        probs = F.softmax(logits, dim=0)  # convert to probabilities
        top2_probs, top2_indices = torch.topk(probs, 2)
        results = [(idx2lang[idx.item()], top2_probs[i].item()*100) for i, idx in enumerate(top2_indices)]
        return results

# Interactive loop
while True:
    input_word = input("Enter a word: ")
    clean_input_word = clean_word(input_word)
    top2 = predict_top2_prob(encode_to_tensor(clean_input_word), model, {i: l for l, i in lang2idx.items()})
    print(f"{clean_input_word} ({input_word}) -> {top2[0][0]} ({top2[0][1]:.1f}%), {top2[1][0]} ({top2[1][1]:.1f}%)")
