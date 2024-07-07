import tiktoken

encoding = tiktoken.encoding_for_model("gpt2")

encoded = encoding.encode("Hello")

print(encoded, encoding.decode(encoded))




import sys; sys.exit(0)
from bpe import BPE, merge
import numpy as np

""" bpe = BPE()
bpe.train_bpe(100, 'taylorswift.txt') """

""" bpe = BPE()
print(bpe.decode(bpe.encode('hello world'))) """


from bpe import BPE

bpe = BPE()
bpe.train_bpe(100, 'taylorswift.txt')


