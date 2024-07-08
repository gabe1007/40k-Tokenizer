from bpe import BPE

bpe = BPE()
bpe.train_bpe(1000, 'Eisenhorn.txt')

""" import numpy as np  

print(np.load('vocab.npy', allow_pickle=True).item()) """