#!/usr/bin/env python

import nltk, zipfile, argparse, re
from nltk.probability import FreqDist
from nltk.probability import ConditionalFreqDist
from contextlib import redirect_stdout

###############################################################################
## Utility Functions ##########################################################
###############################################################################
# This method takes the path to a zip archive.
# It first creates a ZipFile object.
# Using a list comprehension it creates a list where each element contains
# the raw text of the fable file.
# We iterate over each named file in the archive:
#     for fn in zip_archive.namelist()
# For each file that ends with '.txt' we open the file in read only
# mode:
#     zip_archive.open(fn, 'rU')
# Finally, we read the raw contents of the file:
#     zip_archive.open(fn, 'rU').read()
def unzip_corpus(input_file):
    zip_archive = zipfile.ZipFile(input_file)
    contents = [zip_archive.open(fn, 'r').read().decode('utf-8') for fn in zip_archive.namelist() if fn.endswith(".txt")
                and not fn.startswith('__MACOSX')]
    return contents

###############################################################################
## Stub Functions #############################################################
###############################################################################

def process_corpus(corpus_name):
    print(f'1. Corpus name: {corpus_name}')
    input_file = corpus_name + ".zip"
    corpus_contents = unzip_corpus(input_file)
    corpus_sentences = []
    for content in corpus_contents:
        corpus_sentences.append(re.split(r'(?<=\.) ', content))
    corpus_words = []
    allwords = []
    for sent in corpus_sentences:
        words = []
        for word in sent:
            x = nltk.word_tokenize(word)
            words.append(x)
            for w in x:
                allwords.append(w.lower())
        corpus_words.append(words)
    f = open(corpus_name + "-pos.txt", "w")
    allpos = []
    for story in corpus_words:
        for sentence in story:
            sent = nltk.pos_tag(sentence)
            for word in sent:
                f.write(word[0] + "/" + word[1] + " ")
                allpos.append(word)
        f.write("\n\n")
    f.close()
    print(f'\n2. Total words in the corpus: {len(allwords)}')
    numunique = len(set(allwords))
    print(f'\n3. Vocabulary size of the corpus: {numunique}')
    posfreq = {}
    for i in allpos:
        if i[1] in posfreq:
            posfreq[i[1]] += 1
        else:
            posfreq[i[1]] = 1
    inv = {v: k for k, v in posfreq.items()}
    sorted_posfreq = {k: inv[k] for k in sorted(inv)}
    l = list(sorted_posfreq.keys())
    print(f'\n4. The most frequent part-of-speech tag is {sorted_posfreq.get(l[-1])} with frequency {l[-1]}')
    f = open(corpus_name + "-word-freq.txt", "w")
    fdist = FreqDist(word for word in allwords)
    fdist.pprint(maxlen=numunique,stream=f)
    f.close()
    cfdist = ConditionalFreqDist((word[1], word[0].lower()) for word in allpos)
    print(f'\n5. Frequencies and relative frequencies of all part-of-speech tags in the corpus in decreasing order of frequency are: ')
    for i in range(1, len(sorted_posfreq)):
        print(f'{sorted_posfreq.get(l[-i])} tag has frequency {l[-i]} and relative frequency {round(l[-i]/3676, 3)}.')
    f = open(corpus_name + "-pos-word-freq.txt", "w")
    with redirect_stdout(f):
        cfdist.tabulate()
    f.close()
    text = nltk.Text(allwords)
    pos_list = ["NN", "VBD", "JJ", "RB"]
    print("\n6.")
    for pos in pos_list:
        m = cfdist[pos].max()
        print(f'The most frequent word in the POS {pos} is {m} and its most similar words are:')
        text.similar(m)
    print(f'7. Collocations:')
    text.collocations()

###############################################################################
## Program Entry Point ########################################################
###############################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assignment 1')
    parser.add_argument('--corpus', required=True, dest="corpus", metavar='NAME',  help='Which corpus to process {fables, blogs}')

    args = parser.parse_args()
    
    corpus_name = args.corpus
    
    if corpus_name == "fables" or "blogs":
        process_corpus(corpus_name)
    else:
        print("Unknown corpus name: {0}".format(corpus_name))
        