import json
import os
import pickle
import re
import string
import tarfile
from ast import literal_eval
from collections import Counter
from datetime import datetime
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from tqdm import tqdm

most_common_words = ['w', 'z', 'i', 'na', 'do', 'nie', 'k', 'o', 'że', 'art', 'r', 'się', 'przez', 'a', 'dnia', 'od',
                     'sąd', 'jest', 'za', 'oraz']
most_common_trans = ['w:prep', 'z:prep', 'na:prep', 'i:conj', 'do:prep', 'nie:qub', 'on:ppron3', 'dzień:subst',
                     'o:prep',
                     'sąd:subst', 'że:comp', 'ten:adj', 'się:qub', 'przez:prep', 'który:adj', 'od:prep', 'a:conj',
                     'być:fin', 'art:brev', 'rok:brev']

Courts_to_check = ("COMMON", "SUPREME")

group_match = [
    re.compile(r"\bA?C.*\b"),
    re.compile(r"\bA?U.*\b"),
    re.compile(r"\bA?K.*\b"),
    re.compile(r"\bG.*\b"),
    re.compile(r"\bA?P.*\b"),
    re.compile(r"\bR.*\b"),
    re.compile(r"\bW.*\b"),
    re.compile(r"\bAm.*\b")
]
names = ["A?C.*",
         "A?U.*",
         "A?K.*",
         "G.*",
         "A?P.*",
         "R.*",
         "W.*",
         "Am.*", ]

st = r"\W+"
res1 = r"(<[^>]*>)"
res2 = r"(-(\n|\r|\r\n|\x0b|\x0c|\x1c|\x1d|\x1e|\x85|\u2028|\u2029))"
res3 = r"^[a-ząćęłńóśźż]+$"
uzas = r"^[\s\S]*?UZASADNIENIE([\s\S]*)$"
com = re.compile(st, re.IGNORECASE)
com1 = re.compile(res1, re.IGNORECASE)
com2 = re.compile(res2, re.IGNORECASE)
pairs1 = re.compile(res3, re.IGNORECASE)
from_uzas = re.compile(uzas, re.IGNORECASE)

with open("file_list", "rb") as file:
    file_dict = pickle.load(file)

flag = False


def filter_uza(x: str):
    global flag
    flag = flag or x.startswith("uzasadnienie") or x.startswith("uzasadnić")
    return flag


def trans_count(file):
    file_path = os.path.join("../zad5/done", file)

    def inner_work():
        if loaded[0][0].startswith("uzasadnienie"):
            yield loaded[0][0]
            yield from map(lambda x: x[1].lower(), loaded)
        else:
            yield from filter(filter_uza, map(lambda x: x[1].lower(), loaded))

    if os.path.isfile(file_path):
        try:
            with open(file_path, "r") as f:
                loaded = literal_eval((f.read()).replace(")(", "), ("))
            yield from filter(lambda t: t not in most_common_trans, inner_work())
        except SyntaxError as err:
            print(file_path + ":  " + str(err))


def get_next(tars):
    index = 1
    for tarinfo in tars:
        if tarinfo.isreg():
            if tarinfo.name.startswith('data/json/judgments'):
                index += 1
                yield json.load(tars.extractfile(tarinfo)), tarinfo.name[10:]


def given_date(date):
    z = datetime.strptime(date, '%Y-%m-%d')
    return z.year == 2015


def conditions(name, ids, court_type, date):
    return court_type in Courts_to_check and given_date(date) and ids in file_dict[name]


def get_one(text):
    def inner(fun, itera):
        i = filter(lambda word: word not in most_common_words, filter(fun,
                                                                      map(lambda w: w.translate(str.maketrans("", "",
                                                                                                              "§" + string.punctuation)).lower(),
                                                                          itera)))

        for word in i:
            yield word

    yield from inner(pairs1.match, text)


def give_me(tars):
    global flag
    for x, name in get_next(tars):
        for y in x['items']:
            try:
                if conditions(name[:-5], y['id'], y['courtType'], y['judgmentDate']):
                    for num, matching in enumerate(group_match):
                        if matching.search(y['courtCases'][0]['caseNumber']):
                            words = com.split(from_uzas.search(com2.sub("", com1.sub("", y['textContent']))).group(1))

                            if len(words) > 0:
                                # print(name, y['id'])
                                yield num, Counter(get_one(words)), Counter(trans_count(name[:-5] + ":" + str(y["id"])))
                                flag = False
                            break
                            # path = os.path.join("../zad5/done", name[:-5] + ":" + str(y["id"]))
                            # with open(path, "r") as f:
                            #     loaded = eval((f.read()).replace(")(", "), ("))
            except (AttributeError, IndexError, KeyError) as err:
                pass


def to_count_files(tars_path, name="done"):
    tar = tarfile.open(tars_path, "r:gz", )
    for i, res in tqdm(enumerate(give_me(tar))):
        path = os.path.join(name, "norm", str(res[0]), str(i))
        with open(path, "w") as f:
            f.write(str(dict(res[1])))
        path = os.path.join(name, "trans", str(res[0]), str(i))
        with open(path, "w") as f:
            f.write(str(dict(res[2])))
    tar.close()


def load_from_to_countre(paths="done/norm"):
    for directory in os.listdir(paths):
        path = os.path.join(paths, directory)
        for file in os.listdir(path):
            with open(os.path.join(path, file), "r") as f:
                yield int(directory), Counter(literal_eval(f.read()))


def save_to_sparse_bag(path="norm"):
    result = [[] for _ in range(8)]
    final_bag = set()
    for directory, counter in tqdm(load_from_to_countre(paths=os.path.join("done", path), desc="Load")):
        final_bag = final_bag | set(counter)
        result[directory].append(counter)
    bag_train = []
    bag_test = []
    permutation_train = []
    permutation_test = []
    final_bag = sorted(list(final_bag))
    for i, res in tqdm(enumerate(result), desc="Bag"):
        train, test = train_test_split(res, test_size=0.25)
        permutation_test.extend([i] * len(test))
        permutation_train.extend([i] * len(train))
        bag_test.extend(test)
        bag_train.extend(train)
    train_perm = np.random.permutation(len(bag_train))
    test_perm = np.random.permutation(len(bag_test))
    permutation_train = np.array(permutation_train)[train_perm]
    permutation_test = np.array(permutation_test)[test_perm]
    result = []
    with open(os.path.join(path, "train_perm"), "w") as f:
        f.write(str(permutation_train.tolist()))
    with open(os.path.join(path, "test_perm"), "w") as f:
        f.write(str(permutation_test.tolist()))
    bag_test = np.array(bag_test)[test_perm]
    bag_train = np.array(bag_train)[train_perm]
    for name, bag in tqdm((("test", bag_test), ("train", bag_train)), desc="Make"):
        mat = sparse.lil_matrix((len(bag), len(final_bag)))
        for i, count in tqdm(enumerate(bag), desc="Matrix"):
            for j, word in enumerate(final_bag):
                if count[word] != 0:
                    mat[i, j] = count[word]
        sparse.save_npz(os.path.join(path, "sparse_{}.npz".format(name)), mat.tocsr())


def SVM_test(path):
    with open(os.path.join(path, "train_perm"), "r") as f:
        train_ans = literal_eval(f.read())
    with open(os.path.join(path, "test_perm"), "r") as f:
        test_ans = literal_eval(f.read())
    X_test = TfidfTransformer(use_idf=True).fit_transform(sparse.load_npz(os.path.join(path, "sparse_test.npz")))
    X_train = TfidfTransformer(use_idf=True).fit_transform(sparse.load_npz(os.path.join(path, "sparse_train.npz")))
    print(X_test.shape)
    print(X_train.shape)
    for i, name in enumerate(names):
        print(name)
        Y_train = np.array([1 if x == i else 0 for x in train_ans])
        Y_test = np.array([1 if x == i else 0 for x in test_ans])
        cla = SVC(kernel="linear", C=100)
        cla.fit(X_train, Y_train)
        print(classification_report(Y_test, cla.predict(X_test)))

def lin_SVM_test(path):
    with open(os.path.join(path, "train_perm"), "r") as f:
        Y_train = literal_eval(f.read())
    with open(os.path.join(path, "test_perm"), "r") as f:
        Y_test = literal_eval(f.read())
    X_test = TfidfTransformer(use_idf=True).fit_transform(sparse.load_npz(os.path.join(path, "sparse_test.npz")))
    X_train = TfidfTransformer(use_idf=True).fit_transform(sparse.load_npz(os.path.join(path, "sparse_train.npz")))
    print(X_test.shape)
    print(X_train.shape)
    for i, name in enumerate(names):
        print(name)
        cla = LinearSVC(C=100)
        cla.fit(X_train, Y_train)
        Y_pre = cla.predict(X_test)
        print("accuracy_score")
        print(accuracy_score(Y_test, Y_pre))
        print("classification_report")
        print(classification_report(Y_test, Y_pre))
        print("micro_report")
        print(precision_recall_fscore_support(Y_test, Y_pre, average='micro'))
        print("macro_report")
        print(precision_recall_fscore_support(Y_test, Y_pre, average='macro'))


if __name__ == "__main__":
    # to_count_files("../saos-dump-23.02.2018.tar.gz")
    # save_to_sparse_bag("norm")
    # save_to_sparse_bag("trans")
    # SVM_test("norm")
    # SVM_test("trans")
    with open("result_list", "rb") as f:
        classes = pickle.load(f)
    for i, cla in enumerate(classes):
        print("Class {} has {} samples.".format(names[i], len(cla)))
