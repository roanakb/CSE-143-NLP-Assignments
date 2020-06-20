import nltk
from nltk.corpus import wordnet as wn

if __name__ == '__main__':
    w1 = "cat"
    f = open("wordnet.txt", "w", encoding="utf-8")
    f.write(f'Synsets and Definitions of {w1}:\n')
    synsets = wn.synsets(w1)
    for synset in synsets:
        f.write(f'{synset}: {synset.definition()}\n')
    s = synsets[0]
    f.write(f'Hyponyms of {s}:\n{s.hyponyms()}\n')
    f.write(f'Root Hypernyms of {s}:\n{s.root_hypernyms()}\n')
    f.write(f'Hypernyms Path to Root of {s}:\n{s.hypernym_paths()}\n')
    w2 = "banana"
    b_synset = wn.synsets(w2)[0]
    f.write(f'Path similarity from {w1} to {w2} is {s.path_similarity(b_synset):.4f}\n')
    names = ['dog.n.01', 'man.n.01', 'whale.n.01', 'bark.n.01', 'cat.n.01']
    sets = []
    for name in names:
        sets.append(wn.synset(name))
    similarities = {}
    for i in range(0, len(sets)):
        for j in range(i, len(sets)):
            if j is not i:
                t = (i, j)
                similarities[t] = sets[i].path_similarity(sets[j])
                print(f'{i}, {j}, {similarities[t]}')
    max = 0
    solns = []
    for i in range(0, len(similarities)):
        key = list(similarities.keys())[i]
        item = similarities.get(key)
        if item > max:
            max = item
            solns.clear()
            solns.append(key)
        elif item == max:
            solns.append(key)
    f.write(f'Words with highest path similarity from {names}:\n')
    for t in solns:
        f.write(f'{sets[t[0]]} to {sets[t[1]]} with similarity of {similarities[t]:.4f}\n')
    f.close()