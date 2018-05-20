import glob
import json
import re
import tarfile
from datetime import datetime

from gensim.models.phrases import Phrases, Phraser
from gensim.models.word2vec import *
from gensim.utils import SaveLoad
from gensim.utils import tokenize
from tqdm import tqdm

res1 = r"(<[^>]*>)"
res2 = r"(-(\n|\r|\r\n|\x0b|\x0c|\x1c|\x1d|\x1e|\x85|\u2028|\u2029))"

com1 = re.compile(res1, re.IGNORECASE)
com2 = re.compile(res2, re.IGNORECASE)


def get_next(tars):
    index = 1
    for tarinfo in tars:
        if tarinfo.isreg():
            if tarinfo.name.startswith('data/json/judgments'):
                index += 1
                yield json.load(tars.extractfile(tarinfo)), tarinfo.name[10:-5]


def given_date(date):
    z = datetime.strptime(date, '%Y-%m-%d')
    return z.year == 2015


def give_me(tars):
    for x, name in tqdm(get_next(tars)):
        for y in x['items']:
            try:
                if given_date(y['judgmentDate']):
                    words = com2.sub("", com1.sub("", y['textContent']))
                    with open(os.path.join("data", f"{name}:{y['id']}.txt"), "w") as f:
                        f.write(words)
            except (AttributeError, IndexError, KeyError) as err:
                pass


def from_file(paths, lower=True):
    # print(paths)
    for name in tqdm(glob.glob(os.path.join(paths, "*"))):
        # print(name)
        with open(name, "r") as f:
            with open(os.path.join("preprocesssed", name[5:]), "a") as f1:
                for line in f:
                    words = list(tokenize(line, lowercase=lower))
                    # words = list(gensim.utils.simple_tokenize(line))
                    if words:
                        # print(words)
                        # yield words
                        f1.write(" ".join(words))
                        f1.write("\n")
                        # yield list(gensim.models.word2vec.LineSentence(name))


def make_bi_tri(paths, tri=False):
    sentences = PathLineSentences(paths)
    phases = Phrases(sentences)
    bigram = Phraser(phases)
    bigram.save()
    if tri:
        triphases = Phrases(bigram[sentences])
        trigram = Phraser(triphases)
        trigram.save()


if __name__ == "__main__":
    tar = tarfile.open("../saos-dump-23.02.2018.tar.gz", "r:gz")
    give_me(tar)
    tar.close()
    from_file("data", lower=True)
    sentences = PathLineSentences(os.path.join(os.getcwd(), "preprocesssed"))
    bigram = SaveLoad.load("bigram")
    trigram = SaveLoad.load("trigram")
    word = Word2Vec(trigram[bigram[sentences]], window=5, sg=0, size=300, min_count=3, workers=7)
    word.save("counted_model")
