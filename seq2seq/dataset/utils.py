from __future__ import print_function
import unicodedata
import re


def filter_pair(pair, src_max_len, tgt_max_len):
    """
    Returns true if a sentence pair meets the length requirements, false otherwise.

    Args:
        pair ((str, str)): (source, target) sentence pair
        src_max_len (int): maximum length cutoff for sentences in the source language
        tgt_max_len (int): maximum length cutoff for sentences in the target language
    Returns:
         bool: true if the pair is shorter than the length cutoffs, false otherwise
    """
    return len(pair[0]) <= src_max_len and len(pair[1]) <= tgt_max_len


def space_tokenize(text):
    """
    Tokenizes a piece of text by splitting it up based on single spaces (" ").

    Args:
     text (str): input text as a single string

    Returns:
         list(str): list of tokens obtained by splitting the text on single spaces
    """
    return text.split(" ")


def prepare_data(path, src_max_len, tgt_max_len, tokenize_func=space_tokenize):
    """
    Reads a tab-separated data file where each line contains a source sentence and a target sentence. Pairs containing
    a sentence that exceeds the maximum length allowed for its language are not added.

    Args:
        path (str): path to the data file
        src_max_len (int): maximum length cutoff for sentences in the source language
        tgt_max_len (int): maximum length cutoff for sentences in the target language
        tokenize_func (func): function for splitting words in a sentence (default is single-space-delimited)

    Returns:
        list((str, str)): list of (source, target) string pairs
    """

    print("Reading lines...")

    # Read the file and split into lines
    pairs = []
    counter = 0
    with open(path) as fin:
        for line in fin:
            try:
                src, dst = line.strip().split("\t")
                pair = map(lambda st: tokenize_func(st), [src, dst])
                if filter_pair(pair, src_max_len, tgt_max_len):
                    pairs.append(pair)
            except:
                print("Error when reading line: {0}".format(line))
                raise
            counter += 1
            if counter % 100 == 0:
                print("\rRead {0} lines".format(counter), end="")

    print("\nNumber of pairs: %s" % len(pairs))
    return pairs


def read_vocabulary(path, max_num_vocab=50000):
    """
    Helper function to read a vocabulary file.

    Args:
        path (str): filepath to raw vocabulary file
        max_num_vocab (int): maximum number of words to read from vocabulary file

    Returns:
        set: read words from vocabulary file
    """
    print("Reading vocabulary...")

    # Read the file and create list of tokens in vocabulary
    vocab = set()
    with open(path) as fin:
        for line in fin:
            if len(vocab) >= max_num_vocab:
                break
            try:
                vocab.add(line.strip())
            except:
                print ("Error when reading line: {0}".format(line))
                raise

    print("\nSize of Vocabulary: %s" % len(vocab))
    return vocab

