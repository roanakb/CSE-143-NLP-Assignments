import re, nltk, argparse
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.util import ngrams
from nltk.lm import NgramCounter


def get_score(review):
    return int(re.search(r'Overall = ([1-5])', review).group(1))

def get_text(review):
    return re.search(r'Text = "(.*)"', review).group(1)

def read_reviews(file_name):
    """
    Dont change this function.

    :param file_name:
    :return:
    """
    file = open(file_name, "rb")
    raw_data = file.read().decode("latin1")
    file.close()

    positive_texts = []
    negative_texts = []
    first_sent = None
    for review in re.split(r'\.\n', raw_data):
        overall_score = get_score(review)
        review_text = get_text(review)
        if overall_score > 3:
            positive_texts.append(review_text)
        elif overall_score < 3:
            negative_texts.append(review_text)
        if first_sent == None:
            sent = nltk.sent_tokenize(review_text)
            if (len(sent) > 0):
                first_sent = sent[0]
    return positive_texts, negative_texts, first_sent


########################################################################
## Dont change the code above here
######################################################################



def process_reviews(file_name):
    positive_texts, negative_texts, first_sent = read_reviews(file_name)

    # There are 150 positive reviews and 150 negative reviews.
    # print(len(positive_texts))
    # print(len(negative_texts))
    pos = []
    poswords = []
    neg = []
    negwords = []
    for i in range(0, len(positive_texts)):
        p = normalize(word_tokenize(positive_texts[i]))
        for item in p:
            poswords.append(item)
        pos.append(p)
        n = normalize(word_tokenize(negative_texts[i]))
        for item in n:
            negwords.append(item)
        neg.append(n)
    pu = open("POSITIVE-unigram-freq.txt", 'w', encoding="utf-8")
    nu = open("NEGATIVE-unigram-freq.txt", 'w', encoding="utf-8")
    pb = open("POSITIVE-bigram-freq.txt", 'w', encoding="utf-8")
    nb = open("NEGATIVE-bigram-freq.txt", 'w', encoding="utf-8")


    fdist = FreqDist(word for word in poswords)
    print(fdist["the"])
    print(fdist["wine"])
    print(fdist["list"])
    pos_unigrams = [ngrams(sent, 1) for sent in pos]
    pos_bigrams = [ngrams(sent, 2) for sent in pos]
    pos_trigrams = [ngrams(sent, 3) for sent in pos]
    pos_4grams = [ngrams(sent, 4) for sent in pos]
    pos_5grams = [ngrams(sent, 5) for sent in pos]

    pos_counts = NgramCounter(pos_unigrams + pos_bigrams + pos_trigrams + pos_4grams + pos_5grams)

    neg_unigrams = [ngrams(sent, 1) for sent in neg]
    neg_bigrams = [ngrams(sent, 2) for sent in neg]
    neg_trigrams = [ngrams(sent, 3) for sent in neg]
    neg_4grams = [ngrams(sent, 4) for sent in neg]
    neg_5grams = [ngrams(sent, 5) for sent in neg]

    neg_counts = NgramCounter(neg_unigrams + neg_bigrams + neg_trigrams + neg_4grams + neg_5grams)

    p1 = pos_counts[1]
    p2 = pos_counts[2]
    p3 = pos_counts[3]
    p4 = pos_counts[4]
    p5 = pos_counts[5]

    print(fdist.N())
    print(p2[('restaurant', 'excellent')])
    n1 = neg_counts[1]
    n2 = neg_counts[2]
    n3 = neg_counts[3]
    n4 = neg_counts[4]
    n5 = neg_counts[5]

    unigramout(p1, pu)
    unigramout(n1, nu)
    bigramout(p2, pb)
    bigramout(n2, nb)

    postext = nltk.Text(poswords)
    negtext = nltk.Text(negwords)
    postext.collocations()
    negtext.collocations()
    return
    # Your code goes here

def bigramout(dict, file):
    for cond_word in dict:
        g = dict.get(cond_word)
        for word, frequency in g.most_common():
            file.write(f'{cond_word[0]} {word} {frequency}\n')

def unigramout(dict, file):
    result = {}
    for word, frequency in dict.most_common():
        result[frequency] = word
    k = list(result.keys())
    k.sort()
    for key in k:
        file.write(f'{result.get(key)} {key}\n')

def normalize(wordarr):
    # Takes in array of words in sentence and returns normalized array
    regex = re.compile(r'\w')
    stop_words = stopwords.words('english')
    lowered_words = []
    for word in wordarr:
        word = word.lower()
        if re.findall(regex, word) and word not in stop_words:
            lowered_words.append(word)
    return lowered_words

# Write to File, this function is just for reference, because the encoding matters.
def write_file(file_name, data):
    file = open(file_name, 'w', encoding="utf-8")    # or you can say encoding="latin1"
    file.write(data)
    file.close()


def write_unigram_freq(category, unigrams):
    """
    A function to write the unigrams and their frequencies to file.

    :param category: [string]
    :param unigrams: list of (word, frequency) tuples
    :return:
    """
    uni_file = open("{0}-unigram-freq-n.txt".format(category), 'w', encoding="utf-8")
    for word, count in unigrams:
        uni_file.write("{0:<20s}{1:<d}\n".format(word, count))
    uni_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assignment 2')
    parser.add_argument('-f', dest="fname", default="restaurant-training.data",  help='File name.')
    args = parser.parse_args()
    fname = args.fname

    process_reviews(fname)
