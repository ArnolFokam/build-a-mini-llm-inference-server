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
def top_p_filter(logits, p):
    # TODO: keep smallest set of tokens whose cumulative prob >= p, mask the rest to -inf.

    # 1. get the probabilities
    scores = logits - logits.max(axis=-1, keepdims=True)
    scores = np.exp(scores)
    scores /= scores.sum(axis=-1, keepdims=True)

    # 2. sort the probabilites and get the cumsum
    sorted_idx = np.argsort(-scores, axis=-1)
    keep_sorted = np.take_along_axis(scores, sorted_idx, axis=-1)
    summed = np.cumsum(keep_sorted, axis=-1)

    # 3. get the mask used to update the original array
    shifted_summed = np.concatenate(
        [np.zeros_like(scores)[..., :1], summed[..., :-1]],
        axis=-1
    )
    mask = shifted_summed < p

    # 4. map the mask back to the original array
    keep = np.empty_like(keep_sorted)
    np.put_along_axis(keep, sorted_idx, mask, axis=-1)

    return np.where(keep, logits, float('-inf'))

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
    # TODO: build a character-level vocab; specials get the lowest ids, then sorted unique chars.
    build_vocab = {
        'id_to_token': [],
        'token_to_id': {}
    }
    counter = 0

    # start with special_tokens
    for tok in special_tokens:

        if tok not in build_vocab['token_to_id']:
            build_vocab['id_to_token'].append(tok)
            build_vocab['token_to_id'][tok] = counter
            counter += 1

    # continue with the other tokens
    chars = sorted({char for word in corpus for char in word})
    for tok in chars:

        if tok not in build_vocab['token_to_id']:
            build_vocab['id_to_token'].append(tok)
            build_vocab['token_to_id'][tok] = counter
            counter += 1

    return build_vocab

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

