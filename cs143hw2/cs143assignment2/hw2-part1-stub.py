import argparse, re, nltk
from nltk.probability import FreqDist

# https://docs.python.org/3/howto/regex.html
# https://docs.python.org/3/library/re.html
# https://www.debuggex.com/


def get_words(pos_sent):
    """
    Given a part-of-speech tagged sentence, return a sentence
    including each token separated by whitespace.

    As an interim step you need to fill word_list with the
    words of the sentence.

    :param pos_sent: [string] The POS tagged sentence
    :return:
    """

    word_list = []
    word_list = pos_sent.split()
    punc = re.compile(r'[.,]/[,.]')
    i = 0
    while True:
        if (i >= len(word_list)):
            break
        w = word_list[i]
        if punc.match(w):
            word_list.remove(w)
        else:
            word_list[i] = re.sub(r'/[A-Za-z]+', "", w)
            i+=1

    # Write a regular expression that matches only the
    # words of each word/pos-tag in the sentence.

    retval = " ".join(word_list) if len(word_list) > 0 else None
    return retval

def get_pos_tags(pos_sent):
    pos_sent = re.sub(r'[.,;:\n]', "", pos_sent)
    pos_list = re.split(r'[ ]*[a-zA-Z]*/', pos_sent)
    for i in pos_list:
        if i is "":
            pos_list.remove(i)
    return pos_list

def get_noun_phrases(pos_sent):
    """
    Find all simple noun phrases in pos_sent.

    A simple noun phrase is a single optional determiner followed by zero
    or more adjectives ending in one or more nouns.

    This function should return a list of noun phrases without tags.

    :param pos_sent: [string]
    :return: noun_phrases: [list]
    """

    #pos_sent = "The/DT Donkey/NNP and/CC the/DT Mule/NNP A/NNP MULETEER/NNP set/VBD forth/NN on/IN a/WDT journey/NN ,/, driving/VBG before/IN him/PRP a/DT Donkey/NNP and/CC a/DT Mule/NNP ,/, both/DT well/RB laden/NN ./. The/DT Donkey/NNP ,/, as/RB long/RB as/IN he/PRP traveled/VBD along/RP the/DT plain/NN ,/, carried/VBD his/PRP$ load/NN with/IN ease/NN ,/, but/CC when/WRB he/PRP began/VBD to/TO ascend/VB the/DT steep/JJ path/NN of/IN the/DT mountain/NN ,/, felt/VBD his/PRP$ load/NN to/TO be/VB more/JJR than/IN he/PRP could/MD bear/VB ./. He/PRP entreated/VBD his/PRP$ companion/NN to/TO relieve/VB him/PRP of/IN a/DT small/JJ portion/NN ,/, that/IN he/PRP might/MD carry/VB home/NN the/DT rest/NN ;/: but/CC the/DT Mule/NNP paid/VBD no/DT attention/NN to/TO the/DT request/NN ./. The/DT Donkey/NNP shortly/RB afterwards/VBZ fell/VBD down/RB dead/JJ under/IN his/PRP$ burden/NN ./. Not/RB knowing/VBG what/WP else/RB to/TO do/VB in/IN so/RB wild/JJ a/DT region/NN ,/, the/DT Muleteer/NNP placed/VBD upon/IN the/DT Mule/NNP the/DT load/NN carried/VBN by/IN the/DT Donkey/NNP in/IN addition/NN to/TO his/PRP$ own/JJ ,/, and/CC at/IN the/DT top/NN of/IN all/DT placed/VBN the/DT hide/NN of/IN the/DT Donkey/NNP ,/, after/IN he/PRP had/VBD skinned/VBN him/PRP ./. The/DT Mule/NNP ,/, groaning/VBG beneath/IN his/PRP$ heavy/JJ burden/NN ,/, said/VBD to/TO himself/PRP :/: ``/`` I/PRP am/VBP treated/VBN according/VBG to/TO my/PRP$ deserts/NNS ./. If/IN I/PRP had/VBD only/RB been/VBN willing/JJ to/TO assist/VB the/DT Donkey/NNP a/DT little/JJ in/IN his/PRP$ need/NN ,/, I/PRP should/MD not/RB now/RB be/VB bearing/VBG ,/, together/RB with/IN his/PRP$ burden/NN ,/, himself/PRP as/RB well/RB ./. ''/'' "
    # pos_sent = " A/DT big/ADJ Wolf/NNP resolved/VBD to/TO disguise/VB himself/PRP in/IN order/NN that/IN he/PRP might/MD prey/VB upon/IN a/DT flock/NN of/IN sheep/NN without/IN fear/NN of/IN detection/NN ./. So/RB he/PRP clothed/VBD himself/PRP in/IN a/DT sheepskin/NN ,/, and/CC slipped/VBD among/IN the/DT sheep/NN when/WRB they/PRP were/VBD out/RB at/IN pasture/NN ./. He/PRP completely/RB deceived/VBD the/DT shepherd/NN ,/, and/CC when/WRB the/DT flock/NN was/VBD penned/VBN for/IN the/DT night/NN he/PRP was/VBD shut/VBN in/IN with/IN the/DT rest/NN ./. But/CC that/DT very/RB night/NN as/IN it/PRP happened/VBD ,/, the/DT shepherd/NN ,/, requiring/VBG a/DT supply/NN of/IN mutton/NN for/IN the/DT table/NN ,/, laid/VBD hands/NNS on/IN the/DT Wolf/NNP in/IN mistake/NN for/IN a/DT Sheep/NNP ,/, and/CC killed/VBD him/PRP with/IN his/PRP$ knife/NN on/IN the/DT spot/NN ./. "
    noun_phrases = []
    pos_sent = " " + pos_sent
    pos_sent = re.sub("\n", "", pos_sent)
    noun_phrases = re.findall(r'(?:(?:[A-Za-z]+/[W]?DT\s)?(?:[A-Za-z]+/JJ[RS]?\s)*)?\s?(?:[A-Za-z]+/NN[SP]*\s)+', pos_sent)
    #noun_phrases = re.findall(r'( [A-Za-z]+/.*T$)*( [A-Za-z]+/J)*( [A-Za-z]+/N)+', pos_sent)(?:[A-Za-z]+/NN[SP]+\s)+

    phrases = []
    for p in noun_phrases:
        s = ""
        for a in p:
            s = s + a
        phrases.append(s)
    for i in range(0, len(phrases)):
        phrases[i] = get_words(phrases[i]).strip()
    return phrases

def read_stories(fname):
    stories = []
    with open(fname, 'r') as pos_file:
        story = []
        for line in pos_file:
            if line.strip():
                story.append(line)
            else:
                stories.append("".join(story))
                story = []
    return stories

def most_freq_noun_phrase(pos_sent_fname, verbose=True):
    """

    :param pos_sent_fname:
    :return:
    """
    story_phrases = {}
    story_id = 1
    for story in read_stories(pos_sent_fname):
        most_common = []
        # your code starts here
        np = get_noun_phrases(story)
        fdist = FreqDist(phrase.lower() for phrase in np)
        most_common = fdist.most_common(3)
        # do stuff with the story

        # end your code
        if verbose:
            print("The most freq NP in document[" + str(story_id) + "]: " + str(most_common))
        story_phrases[story_id] = most_common
        story_id += 1
    return story_phrases

def most_freq_pos_tags(pos_sent_fname, verbose=True):
    """

    :param pos_sent_fname:
    :return:
    """
    story_tags = {}
    story_id = 1
    for story in read_stories(pos_sent_fname):
        most_common = []
        # your code starts here
        pos_tags = get_pos_tags(story)
        fdist = FreqDist(tag for tag in pos_tags)
        # do stuff with the story
        most_common = fdist.most_common(3)
        # end your code
        if verbose:
            print("The most freq pos tags in document[" + str(story_id) + "]: " + str(most_common))
        story_tags[story_id] = most_common
        story_id += 1

    return story_tags



def test_get_words():
    """
    Tests get_words().
    Do not modify this function.
    :return:
    """
    print("\nTesting get_words() ...")
    pos_sent = 'All/DT animals/NNS are/VBP equal/JJ ,/, but/CC some/DT ' \
               'animals/NNS are/VBP more/RBR equal/JJ than/IN others/NNS ./.'
    print(pos_sent)
    retval = str(get_words(pos_sent))
    print("retval:", retval)

    gold = "All animals are equal but some animals are more equal than others"
    assert retval == gold, "test Fail:\n {} != {}".format(retval, gold)

    print("Pass")


def test_get_pos_tags():
    """
    Tests get_pos_tags().
    Do not modify this function.
    :return:
    """
    print("\nTesting get_pos_tags() ...")
    pos_sent = 'All/DT animals/NNS are/VBP equal/JJ ,/, but/CC some/DT ' \
               'animals/NNS are/VBP more/RBR equal/JJ than/IN others/NNS ./.'
    print(pos_sent)
    retval = str(get_pos_tags(pos_sent))
    print("retval:", retval)

    gold = str(['DT', 'NNS', 'VBP', 'JJ', 'CC', 'DT', 'NNS', 'VBP', 'RBR', 'JJ', 'IN', 'NNS'])
    assert retval == gold, "test Fail:\n {} != {}".format(retval, gold)

    print("Pass")



def test_get_noun_phrases():
    """
    Tests get_noun_phrases().
    Do not modify this function.
    :return:
    """
    print("\nTesting get_noun_phrases() ...")

    pos_sent = 'All/DT animals/NNS are/VBP equal/JJ ,/, but/CC some/DT ' \
               'animals/NNS are/VBP more/RBR equal/JJ than/IN others/NNS ./.'
    print("input:", pos_sent)
    retval = str(get_noun_phrases(pos_sent))
    print("retval:", retval)

    gold = "['All animals', 'some animals', 'others']"
    assert retval == gold, "test Fail:\n {} != {}".format(retval, gold)

    print("Pass")


def test_most_freq_noun_phrase(infile="fables-pos.txt"):
    """
    Tests get_noun_phrases().
    Do not modify this function.
    :return:
    """
    print("\nTesting most_freq_noun_phrase() ...")

    import os
    if os.path.exists(infile):
        noun_phrase = most_freq_noun_phrase(infile, False)
        gold1 = "[('the donkey', 6), ('the mule', 3), ('load', 2)]"
        gold2 = "[('the donkey', 6), ('the mule', 3), ('burden', 2)]"
        retval = str(noun_phrase[7])

        print("gold:\t", gold1)
        print("OR:\t", gold2)
        print("retval:\t", retval)

        assert retval == gold1 or retval == gold2, "test Fail:\n {} != {} OR {}".format(noun_phrase[7], gold1, gold2)
        print("Pass")
    else:
        print("Test fail: path does not exist;", infile)

def test_most_freq_pos_tags(infile="fables-pos.txt"):
    """
    Tests most_freq_pos_tags().
    Do not modify this function.
    :return:
    """
    print("\nTesting most_freq_pos_tags() ...")

    import os
    if os.path.exists(infile):
        pos_tags = most_freq_pos_tags(infile, False)
        gold = "[('DT', 28), ('NN', 24), ('IN', 21)]"
        retval = str(pos_tags[7])

        print("gold:\t", gold)
        print("retval:\t", retval)

        assert retval == gold, "test Fail:\n {} != {}".format(pos_tags[7], gold)
        print("Pass")
    else:
        print("Test fail: path does not exist;", infile)


def run_tests():
    test_get_words()
    test_get_pos_tags()
    test_get_noun_phrases()
    test_most_freq_noun_phrase()
    test_most_freq_pos_tags()


if __name__ == '__main__':

    # comment this out if you dont want to run the tests
    run_tests()

    parser = argparse.ArgumentParser(description='Assignment 2')
    parser.add_argument('-i', dest="pos_sent_fname", default="blogs-pos.txt",  help='File name that contant the POS.')

    args = parser.parse_args()
    pos_sent_fname = args.pos_sent_fname

    most_freq_noun_phrase(pos_sent_fname)

