
import nltk
import re
import word_category_counter
import data_helper
import os, sys
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import opinion_lexicon
from nltk.util import ngrams
from nltk.probability import FreqDist


DATA_DIR = "data"
LIWC_DIR = "liwc"

word_category_counter.load_dictionary(LIWC_DIR)


def normalize(token, should_normalize=True):
    """
    This function performs text normalization.

    If should_normalize is False then we return the original token unchanged.
    Otherwise, we return a normalized version of the token, or None.

    For some tokens (like stopwords) we might not want to keep the token. In
    this case we return None.

    :param token: str: the word to normalize
    :param should_normalize: bool
    :return: None or str
    """
    if not should_normalize:
        normalized_token = token
    else:
        regex = re.compile(r'\w')
        stop_words = stopwords.words('english')
        token = token.lower()
        if re.findall(regex, token) and token not in stop_words:
            normalized_token = token
        else:
            normalized_token = None
    return normalized_token



def get_words_tags(text, should_normalize=True):
    """
    This function performs part of speech tagging and extracts the words
    from the review text.

    You need to :
        - tokenize the text into sentences
        - word tokenize each sentence
        - part of speech tag the words of each sentence

    Return a list containing all the words of the review and another list
    containing all the part-of-speech tags for those words.

    :param text:
    :param should_normalize:
    :return:
    """
    words = []
    tags = []

    # tokenization for each sentence
    text = word_tokenize(text)
    pos_tags = nltk.pos_tag(text)
    for pair in pos_tags:
        word = normalize(pair[0], should_normalize)
        if word:
            words.append(word)
            tags.append(pair[1])

    return words, tags


def get_ngram_features(tokens):
    """
    This function creates the unigram and bigram features as described in
    the assignment3 handout.

    :param tokens:
    :return: feature_vectors: a dictionary values for each ngram feature
    """
    feature_vectors = {}
    unigrams = ngrams(tokens, 1)
    bigrams = ngrams(tokens, 2)
    trigrams = ngrams(tokens, 3)
    unigram_dist = FreqDist(word for word in unigrams)
    bigram_dist = FreqDist(word for word in bigrams)
    trigram_dist = FreqDist(word for word in trigrams)

    for item in unigram_dist:
        itemd = f'UNI_{item}'
        feature_vectors[itemd] = unigram_dist.freq(item)
    for item in bigram_dist:
        itemd = f'BIGRAM_{item}'
        feature_vectors[itemd] = bigram_dist.freq(item)
    for item in trigram_dist:
        itemd = f'TRIGRAM_{item}'
        feature_vectors[itemd] = trigram_dist.freq(item)

    return feature_vectors


def get_pos_features(tags):
    """
    This function creates the unigram and bigram part-of-speech features
    as described in the assignment3 handout.

    :param tags: list of POS tags
    :return: feature_vectors: a dictionary values for each ngram-pos feature
    """
    feature_vectors = {}

    unigrams = ngrams(tags, 1)
    bigrams = ngrams(tags, 2)
    trigrams = ngrams(tags, 3)
    unigram_dist = FreqDist(word for word in unigrams)
    bigram_dist = FreqDist(word for word in bigrams)
    trigram_dist = FreqDist(word for word in trigrams)

    for item in unigram_dist:
        itemd = f'UNI_{item}'
        feature_vectors[itemd] = unigram_dist.freq(item)
    for item in bigram_dist:
        itemd = f'BIGRAM_{item}'
        feature_vectors[itemd] = bigram_dist.freq(item)
    for item in trigram_dist:
        itemd = f'TRIGRAM_{item}'
        feature_vectors[itemd] = trigram_dist.freq(item)

    return feature_vectors



def get_liwc_features(words):
    """
    Adds a simple LIWC derived feature

    :param words:
    :return:
    """

    # TODO: binning

    feature_vectors = {}
    newwords = []
    for word in words:
        if word:
            newwords.append(word)
    text = " ".join(newwords)
    liwc_scores = word_category_counter.score_text(text)

    # All possible keys to the scores start on line 269
    # of the word_category_counter.py script
    negative_score = liwc_scores["Negative Emotion"]
    positive_score = liwc_scores["Positive Emotion"]
    anx_score = liwc_scores["Anxiety"]
    sad_score = liwc_scores["Sadness"]
    mad_score = liwc_scores["Anger"]
    cog_score = liwc_scores["Cognitive Processes"]
    per_score = liwc_scores["Perceptual Processes"]
    feature_vectors["Negative Emotion"] = negative_score
    feature_vectors["Positive Emotion"] = positive_score
    feature_vectors["Anxiety"] = anx_score
    feature_vectors["Sad"] = sad_score
    feature_vectors["Angry"] = mad_score
    feature_vectors["Thought"] = cog_score
    feature_vectors["Feel"] = per_score

    if positive_score > negative_score:
        feature_vectors["liwc:positive"] = 1
    else:
        feature_vectors["liwc:negative"] = 1

    return feature_vectors


FEATURE_SETS = {"word_pos_features", "word_features", "word_pos_liwc_features", "word_pos_opinion_features"}


def get_opinion_features(words):
    """
    This function creates the opinion lexicon features
    as described in the assignment3 handout.

    the negative and positive data has been read into the following lists:
    * neg_opinion
    * pos_opinion

    if you haven't downloaded the opinion lexicon, run the following commands:
    *  import nltk
    *  nltk.download('opinion_lexicon')

    :param tags: tokens
    :return: feature_vectors: a dictionary values for each opinion feature
    """
    neg_opinion = opinion_lexicon.negative()
    pos_opinion = opinion_lexicon.positive()
    feature_vectors = {}

    for word in neg_opinion:
        if word in words:
            feature_vectors[word] = 1
        else:
            feature_vectors[word] = 0
    for word in pos_opinion:
        if word in words:
            feature_vectors[word] = 1
        else:
            feature_vectors[word] = 0

    return feature_vectors


def get_features_category_tuples(category_text_dict, feature_set):
    """

    You will might want to update the code here for the competition part.

    :param category_text_dict:
    :param feature_set:
    :return:
    """
    features_category_tuples = []
    all_texts = []

    assert feature_set in FEATURE_SETS, "unrecognized feature set:{}, Accepted values:{}".format(feature_set, FEATURE_SETS)


    for category in category_text_dict:
        for text in category_text_dict[category]:

            words, tags = get_words_tags(text)
            feature_vectors = {}
            if feature_set is "word_pos_features":
                feature_vectors = get_ngram_features(words)
                f = get_pos_features(tags)
                for vect in f:
                    feature_vectors[vect] = f.get(vect)
            elif feature_set is "word_features":
                feature_vectors = get_ngram_features(words)
            elif feature_set is "word_pos_liwc_features":
                feature_vectors = get_liwc_features(words)
            elif feature_set is "word_pos_opinion_features":
                feature_vectors = get_opinion_features(words)
            features_category_tuples.append((feature_vectors, category))
            all_texts.append(text)

    return features_category_tuples, all_texts


def write_features_category(features_category_tuples, outfile_name):
    """
    Save the feature values to file.

    :param features_category_tuples:
    :param outfile_name:
    :return:
    """
    with open(outfile_name, "w", encoding="utf-8") as fout:
        for (features, category) in features_category_tuples:
            fout.write(f'{category} ')
            for feature in features:
                fout.write(f'{feature}: {features.get(feature):1.5f} ')
            fout.write("\n")


def features_stub():
    datafiles = ["imdb-training.data", "imdb-testing.data", "imdb-development.data"]
    featuresets = ["word_pos_features", "word_features", "word_pos_liwc_features", "word_pos_opinion_features"]
    for datafile in datafiles:
        for feature_set in featuresets:
            raw_data = data_helper.read_file(datafile)
            positive_texts, negative_texts = data_helper.get_reviews(raw_data)

            category_texts = {"positive": positive_texts, "negative": negative_texts}

            features_category_tuples, texts = get_features_category_tuples(category_texts, feature_set)
            data_set = re.search(r'-[a-z]+', datafile).group()
            filename = f'{feature_set}{data_set}-features.txt'
            write_features_category(features_category_tuples, filename)



if __name__ == "__main__":
    features_stub()
