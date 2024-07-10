# This is the train file, you just have to input your text 
# and the number of tokens you want to train the tokenizer with.
from bpe import BPE

bpe = BPE()
bpe.train_bpe(1000, 'Eisenhorn.txt')