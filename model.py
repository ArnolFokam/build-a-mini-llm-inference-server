"""
Build a Mini LLM Inference Server

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - stable_softmax
def stable_softmax(x: np.ndarray) -> np.ndarray:
    """Softmax over the last axis, computed stably.

    Subtracts the per-row max before exponentiating so large logits cannot overflow.
    """
    exps = np.exp(x - x.max(axis=-1, keepdims=True))
    return exps / exps.sum(axis=-1, keepdims=True)

# Step 2 - apply_temperature
def apply_temperature(logits, temperature):
    # TODO: scale logits by 1 / temperature; if temperature <= 0, return logits unchanged (greedy).
    if temperature <= 0: return logits

    return logits / temperature

# Step 3 - top_k_filter
def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """Keep the k largest logits per row and set the rest to -inf.

    Accepts shape (V,) or (B, V) and returns the same shape and dtype. If k >= V,
    the logits are returned unchanged. Kept entries retain their original values.
    """
    logits = logits.copy()

    if k >= logits.shape[-1]:
        return logits

    kth_largest = np.partition(logits, -k, axis=-1)[..., [-k]]
    logits = np.where(logits >= kth_largest, logits, -np.inf)
    return logits

# Step 4 - top_p_filter
def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """Nucleus filter: keep the smallest set of tokens whose probability mass is >= p.

    Accepts shape (V,) or (B, V) and filters along the last axis. Tokens are
    ranked by probability (highest first); the smallest prefix whose cumulative
    probability reaches p is kept and every other entry is set to -inf. Kept
    entries retain their original logit values. p is in (0, 1]; p == 1 keeps
    everything, and the top token is always kept.
    """
    probs = stable_softmax(logits)
    sorted_idx = np.argsort(-probs, axis=-1, kind="stable")
    sorted_probs = np.take_along_axis(probs, sorted_idx, axis=-1)

    # Keep a token while the mass ranked above it is still below p.
    mass_before = np.cumsum(sorted_probs, axis=-1) - sorted_probs
    sorted_keep = mass_before < p

    keep = np.zeros_like(sorted_keep)
    np.put_along_axis(keep, sorted_idx, sorted_keep, axis=-1)
    return np.where(keep, logits, -np.inf)

# Step 5 - sample_from_probs
def sample_from_probs(probs, rng):
    # TODO: draw a single token id from the categorical distribution probs using rng
    return rng.choice(len(probs), p=probs)

# Step 6 - greedy_select
def greedy_select(logits):
    # TODO: return the index of the maximum logit (ties -> lowest index).
    return np.argmax(logits)

# Step 7 - build_vocab
def build_vocab(corpus, special_tokens):
    vocab = {}
    unique_tokens = set(special_tokens + corpus)
    vocab["id_to_token"] = {idx: token for idx, token in enumerate(unique_tokens)}
    vocab["token_to_id"] = {token: idx for idx, token in vocab["id_to_token"].items()}
    return vocab

# Step 8 - encode_prompt
def encode_prompt(text, vocab, add_bos=True):
    # TODO: encode text into token ids using vocab, optionally prepending <bos>.
    code = []

    if add_bos:
        code.append(vocab['token_to_id']['<bos>'])

    for c in text:

        if c not in vocab['token_to_id']:
            code.append(vocab['token_to_id']['<unk>'])
        else:
            code.append(vocab['token_to_id'][c])
            
    # code.append(vocab['token_to_id']['<eos>'])
    return code

# Step 9 - decode_tokens
def decode_tokens(token_ids, vocab, skip_special=True):
    # TODO: convert token ids back into a string using vocab['id_to_token'], optionally skipping specials.
    text = ""

    def is_special(token):
        return token.startswith('<') and token.endswith('>')

    for idx in token_ids:

        c = vocab['id_to_token'][idx]

        if is_special(c) and skip_special:
            continue

        text += c

    return text

# Step 10 - embed_tokens
import numpy as np

def embed_tokens(token_ids, embedding_matrix):
    # TODO: return the (T, D) embedding rows for each token id in token_ids
    return np.take(embedding_matrix, token_ids, axis=0)

# Step 11 - linear_projection
def linear_projection(x, weight, bias=None):
    # TODO: Apply y = x @ weight + bias, with bias optional and broadcasting over leading axes.
    y = x @ weight

    if bias is not None:
        y += bias 

    return y

# Step 12 - init_kv_cache
import numpy as np

def init_kv_cache(max_seq_len, d_model):
    # TODO: allocate empty K and V buffers and a length counter for a single sequence
    return {
        'K': np.zeros((max_seq_len, d_model), dtype=np.float32),
        'V': np.zeros((max_seq_len, d_model), dtype=np.float32),
        'length': 0
    }

# Step 13 - append_kv
import numpy as np

def append_kv(cache, k_new, v_new):
    # TODO: write k_new and v_new into the cache starting at cache['length'] and bump length.
    start_idx = cache['length'] - 1
    length = k_new.shape[0]
    cache['K'][start_idx:start_idx + length, :] = k_new
    cache['V'][start_idx:start_idx + length, :] = v_new
    cache['length'] += length
    return cache

