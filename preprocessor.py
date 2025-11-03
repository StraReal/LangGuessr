import csv

def find_language_blocks(dataset, key="639-3"):
    """Return {lang: (start, end)} where each block starts and ends."""
    blocks = {}
    if not dataset:
        return blocks

    current_lang = dataset[0][key]
    start = 0

    for i, entry in enumerate(dataset):
        lang = entry[key]
        if lang != current_lang:
            print(f"→ New language block: {lang}, previous language: {current_lang} (pos {i})")
            blocks[current_lang] = (start, i)
            current_lang = lang
            start = i

    # Add the last block
    blocks[current_lang] = (start, len(dataset))
    return blocks


def save_index_to_csv(blocks, path="indexing.csv"):
    """Save {lang: (start, end)} to a CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["lang", "start", "end"])
        for lang, (start, end) in blocks.items():
            writer.writerow([lang, start, end])
    print(f"✅ Saved {len(blocks)} language blocks to {path}")


def preprocess_dataset(dataset, key="639-3", output="indexing.csv"):
    blocks = find_language_blocks(dataset, key)
    save_index_to_csv(blocks, output)
    return blocks

from datasets import load_dataset
dataset = load_dataset("lbourdois/panlex")["train"]
print('Loaded')

preprocess_dataset(dataset)
