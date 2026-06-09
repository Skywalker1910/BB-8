# Tokenization in BB8

## What is Tokenization?

Tokenization is the process of converting raw text into a sequence of discrete symbols (tokens) that a neural network can process.  Language models operate on sequences of integers, not strings — tokenization is the bridge.

```
Raw text:  "Hello, world!"
              ↓  tokenize
Token IDs: [72, 101, 108, 108, 111, 44, 32, 119, 111, 114, 108, 100, 33]
              ↓  decode
Raw text:  "Hello, world!"
```

The choice of tokenization strategy fundamentally affects:
- **Vocabulary size** — how many unique tokens exist
- **Sequence length** — how many tokens a given text produces
- **OOV handling** — how unknown/rare words are dealt with
- **Computational cost** — longer sequences = more memory and compute

---

## Strategy 1 — Character-Level Tokenization

Every unique character is a token.

```
"Hello" → ['H', 'e', 'l', 'l', 'o'] → [8, 9, 10, 10, 11]
```

**Implemented in:** `tokenizer/char_tokenizer.py`

### Algorithm
1. Collect all unique characters in the corpus
2. Build a vocabulary: `{char: id}` (sorted, special tokens first)
3. Encode: map each character to its ID
4. Decode: join characters back into a string

### Characteristics

| Property | Value |
|---|---|
| Typical vocab size (English text) | 60–100 |
| Sequence length multiplier | 1 token per character |
| OOV tokens | None (any character can be a new token) |
| Training data needed | Very little |

### Advantages
- Zero out-of-vocabulary problem
- Extremely small embedding table

### Disadvantages
- Very long sequences (1 token = 1 character)
- Model must learn word structure from scratch
- High perplexity numerically

---

## Strategy 2 — Word-Level Tokenization

Every word (split on whitespace + punctuation) is a token.

```
"Hello world!" → ['hello', 'world', '!'] → [342, 1089, 5]
```

**Implemented in:** `tokenizer/word_tokenizer.py`

### Algorithm
1. Tokenise text using regex `\w+|[^\w\s]`
2. Count token frequencies
3. Keep the *N* most frequent tokens; everything else → `<UNK>`
4. Encode: map each word to its ID (or `<UNK>`)
5. Decode: join tokens with spaces

### Characteristics

| Property | Value |
|---|---|
| Typical vocab size (English text) | 10,000–100,000 |
| Sequence length | ~1 token per word |
| OOV tokens | All rare/novel words |

### Advantages
- Short sequences
- Directly interpretable tokens

### Disadvantages
- Large vocabulary needed
- Cannot handle new or misspelled words
- Morphological variants ('run', 'running', 'ran') are separate tokens

---

## Strategy 3 — Byte Pair Encoding (BPE)

BPE learns a vocabulary of subword units that balances vocabulary size against sequence length.

```
"lower" → ['low', 'er</w>'] → [423, 67]  (example)
```

**Implemented in:** `tokenizer/bpe_tokenizer.py`

**Paper:** Sennrich et al. (2016). "Neural Machine Translation of Rare Words with Subword Units." ACL.

### Algorithm

**Training phase:**
1. Start with character-level vocabulary, each word split into characters + `</w>` marker
2. Count all adjacent symbol pairs in the corpus
3. Merge the most frequent pair into a single new token
4. Repeat steps 2–3 until target vocabulary size is reached

**Encoding phase:**
1. Split text into words
2. For each word, apply the learned merge rules in order
3. Return the resulting subword token IDs

**Example (simplified):**
```
Initial: l o w </w>       l o w e r </w>       n e w e s t </w>
         (freq=5)         (freq=2)              (freq=6)

Step 1: most frequent pair = ('e', 's')
        → es token added
        → newest</w> becomes: n ew es t </w>   (typo-demo only)

...continue until vocab_size reached
```

### Characteristics

| Property | Value |
|---|---|
| Typical vocab size | 1,000–50,000 (configurable) |
| Sequence length | Between char-level and word-level |
| OOV tokens | Rare (falls back to character pieces) |

### Advantages
- Controllable vocabulary size
- Handles unseen words via character fallback
- Good compression vs. sequence length tradeoff
- Used in GPT-2, GPT-3, RoBERTa, LLaMA

### Disadvantages
- Training is slower (O(V × corpus_size))
- Merge order must be saved and applied consistently

---

## Comparison Table

| | Char | Word | BPE |
|---|---|---|---|
| Vocab size | Tiny (60–100) | Large (10k–100k) | Medium (1k–50k) |
| Sequence length | Very long | Short | Medium |
| OOV problem | None | Severe | Minor |
| Implementation complexity | Simple | Simple | Moderate |
| Used in production LLMs | Rarely | No | Yes (GPT-2, LLaMA) |

---

## Special Tokens

All BB8 tokenizers reserve four special tokens at fixed IDs:

| Token | ID | Purpose |
|---|---|---|
| `<PAD>` | 0 | Padding to equal-length batches |
| `<UNK>` | 1 | Unknown / out-of-vocabulary |
| `<BOS>` | 2 | Beginning of sequence |
| `<EOS>` | 3 | End of sequence |

---

## Compression Efficiency

Compression ratio = characters / tokens.  Higher is better (fewer tokens for the same text).

For the Tiny Shakespeare dataset:

| Tokenizer | Vocab Size | Tokens | Compression |
|---|---|---|---|
| Character | ~65 | ~1,115,394 | 1.0× |
| Word (5k vocab) | 5,000 | ~230,000 | ~4.8× |
| BPE (3k vocab) | 3,000 | ~450,000 | ~2.5× |
| BPE (5k vocab) | 5,000 | ~380,000 | ~2.9× |

*(Approximate figures — run `notebooks/01_tokenization_demo.ipynb` for exact measurements.)*
