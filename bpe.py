from collections import Counter
import numpy as np
import regex as re

def map_func(byte):
    return Counter(zip(byte, byte[1:])) if len(byte) > 1 else Counter()

def get_pair_counts(byte_chunks):
    total_counts = Counter()
    for chunk in byte_chunks:
        total_counts.update(map_func(chunk))
    return total_counts

def merge(new_byte, chunk, pair):
    i = 0
    new_chunk = []
    while i < len(chunk):
        if chunk[i] == pair[0] and i < len(chunk) - 1 and chunk[i + 1] == pair[1]:
            new_chunk.append(new_byte)
            i += 2
        else:
            new_chunk.append(chunk[i])
            i += 1
    return new_chunk

class BPE:

    def __init__(self):
        pass

    def train_bpe(self, vocab_size, text):
        with open(text) as f:
            lines = f.read()

        re_pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        compiled_pattern = re.compile(re_pattern, re.UNICODE)
        chunks = re.findall(compiled_pattern, lines)
        byte_chunks = [list(chunk.encode("utf-8")) for chunk in chunks]

        vocab = {value: chr(value) for value in range(256)}

        merges = {}
        tot = 0
        for i in range(vocab_size):
            pair_counts = get_pair_counts(byte_chunks)
            if pair_counts:
                max_pair = max(pair_counts, key=pair_counts.get)
                new_byte = 256 + i
                byte_chunks = [merge(new_byte, chunk, max_pair) for chunk in byte_chunks]
                merges[max_pair] = new_byte
                vocab[new_byte] = vocab[max_pair[0]] + vocab[max_pair[1]]

        np.save('merges.npy', merges)
        np.save('vocab.npy', vocab)

    def encode(self, sample_text):
        merges = np.load('merges.npy', allow_pickle=True).item()

        bytes_list = list(sample_text.encode('utf-8'))

        i = 0
        while i < len(bytes_list) - 1:
            pair = (bytes_list[i], bytes_list[i+1])
            if pair in merges:
                new_token = merges[pair]
                bytes_list[i] = new_token
                bytes_list.pop(i+1)
                # Move back one step to check for further merges
                i = max(0, i-1)
            else:
                i += 1
    
        return bytes_list
    
    def decode(self, encoded_text):
        vocab = np.load('vocab.npy', allow_pickle=True).item()

        decoded = ''.join([vocab[idx] for idx in encoded_text])
        return decoded