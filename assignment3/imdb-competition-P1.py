
import re, nltk, pickle, argparse
import os
import data_helper
from contextlib import redirect_stdout
from features import get_features_category_tuples

DATA_DIR = "data"


def write_features_category(features_category_tuples, output_file_name):
    output_file = open("{}-features.txt".format(output_file_name), "w", encoding="utf-8")
    for (features, category) in features_category_tuples:
        output_file.write("{0:<10s}\t{1}\n".format(category, features))
    output_file.close()


def get_classifier(classifier_fname):
    classifier_file = open(classifier_fname, 'rb')
    classifier = pickle.load(classifier_file)
    classifier_file.close()
    return classifier


def save_classifier(classifier, classifier_fname):
    classifier_file = open(classifier_fname, 'wb')
    pickle.dump(classifier, classifier_file)
    classifier_file.close()
    info_file = open(classifier_fname.split(".")[0] + '-informative-features.txt', 'w', encoding="utf-8")
    for feature, n in classifier.most_informative_features(100):
        info_file.write("{0}\n".format(feature))
    info_file.close()


def evaluate(classifier, features_category_tuples, reference_text, outfile, data_set_name=None):

    ###     YOUR CODE GOES HERE
    # TODO: evaluate your model
    accuracy = nltk.classify.accuracy(classifier, features_category_tuples)
    features = []
    classes = []
    guesses = []
    for f in features_category_tuples:
        features.append(f[0])
        classes.append(f[1])
        guesses.append(classifier.classify(f[0]))
    probability = classifier.prob_classify_many(features)
    confusion_matrix = nltk.ConfusionMatrix(classes, guesses)

    with open(outfile, "w", encoding="utf-8") as fout:
        with redirect_stdout(fout):
            print(accuracy)
            for pdist in probability:
                print('%.4f %.4f' % (pdist.prob('positive'), pdist.prob('negative')))
            print(confusion_matrix)
    return accuracy, probability, confusion_matrix


def build_features(data_file, feat_name, save_feats=None, binning=False):
    # read text data
    raw_data = data_helper.read_file(data_file)
    positive_texts, negative_texts = data_helper.get_reviews(raw_data)

    category_texts = {"positive": positive_texts, "negative": negative_texts}

    # build features
    features_category_tuples, texts = get_features_category_tuples(category_texts, feat_name)

    # save features to file
    if save_feats is not None:
        write_features_category(features_category_tuples, save_feats)

    return features_category_tuples, texts



def train_model(datafile, feature_set, save_model=None):

    features_data, texts = build_features(datafile, feature_set)

    ###     YOUR CODE GOES HERE
    # TODO: train your model here
    classifier = nltk.NaiveBayesClassifier.train(features_data)
    name = f'imdb-{feature_set}-model-P1.pickle'
    f = open(name, 'wb')
    pickle.dump(classifier, f)
    f.close()
    data_set = re.search(r'-[a-z]+', datafile).group()
    outfile_name = f'{feature_set}{data_set}-informative-features.txt'
    with open(outfile_name, "w", encoding="utf-8") as fout:
        with redirect_stdout(fout):
            classifier.show_most_informative_features()

    if save_model is not None:
        save_classifier(classifier, save_model)
    return classifier


def train_eval(train_file, feature_set, eval_file=None):

    # train the model
    split_name = "train"
    model = train_model(train_file, feature_set)
    #model.show_most_informative_features(20)

    # save the model
    if model is None:
        model = get_classifier(classifier_fname)
    outfile = ""
    if feature_set is "word_features":
        outfile = "output-ngrams.txt"
    elif feature_set is "word_pos_features":
        outfile = "output-pos.txt"
    elif feature_set is "word_pos_liwc_features":
        outfile = "output-liwc.txt"
    elif feature_set is "word_pos_opinion_features":
        outfile = "output-opinion.txt"
    # evaluate the model
    if eval_file is not None:
        features_data, texts = build_features(eval_file, feature_set, binning=None)
        accuracy, probability, cm = evaluate(model, features_data, texts, outfile, data_set_name=None)
        print("The accuracy of {} is: {}".format(eval_file, accuracy))
        print("Proabability per class:")
        print(str(probability))
        print("Confusion Matrix:")
        print(str(cm))
    else:
        accuracy = None

    return accuracy, cm


def main():


    # add the necessary arguments to the argument parser
    parser = argparse.ArgumentParser(description='Assignment 3')
    parser.add_argument('-d', dest="data_fname", default="imdb-training.data",
                        help='File name of the testing data.')
    args = parser.parse_args()


    train_data = args.data_fname


    eval_data = "imdb-development.data"


    for feat_set in ["word_features", "word_pos_features", "word_pos_liwc_features", "word_pos_opinion_features"]:
        print("\nTraining with {}".format(feat_set))
        acc, cm = train_eval(train_data, feat_set, eval_file=eval_data)


    with open("all-results.txt", "w", encoding="utf-8") as fout:
        for feat_set in ["word_features", "word_pos_features"]:
            for data in ["imdb-development.data", "imdb-testing.data"]:
                acc, cm = train_eval(data, feat_set, eval_data)
                fout.write(f'For feature set {feat_set} and data {data}:\nAccuracy: {acc} Confusion Matrix: {cm}\n')








if __name__ == "__main__":
    main()




