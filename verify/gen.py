from gensim.models.fasttext import FastText
from gensim.test.utils import get_tmpfile
from gensim.test.utils import datapath
import os

corpus_file = datapath(os.path.abspath('data/data2_cleaned.txt'))

model = FastText(size=50, window=3, min_count=1)
model.build_vocab(corpus_file=corpus_file)

total_words = model.corpus_total_words

model.train(corpus_file=corpus_file, total_words=total_words, epochs=5)

#fname = get_tmpfile('fasttext.model')
fname = os.path.abspath('fasttext.model')
model.save(fname)

model = FastText.load(fname)

print(model.wv['modi'])
print(model.wv['hfkldsjhkjsdhf'])
