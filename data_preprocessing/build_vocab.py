import json
import string
import nltk
import pickle
import argparse

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(json_path, count_thr):
    with open(json_path) as json_file:
        json_data = json.load(json_file)
    counts = {}
    for jimg in json_data:
        for sent in jimg['paragraph']:
            #txt = str(cap).lower().translate(str.maketrans("", "", string.punctuation)).strip().split()
            tokens = nltk.tokenize.word_tokenize(sent.lower())
            for word in tokens:
                counts[word] = counts.get(word, 0) + 1

    words = [w for w, n in counts.items() if n > count_thr]  #reject words that occur less than 'word count threshhold'

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    vocab.add_word('.')
    """
    tags = ['normal', 'fracture', 'implant', 'tumor', 'osteoarthritis']
    for word in tags:
        if word not in words:
            vocab.add_word(word)
    """
    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

def main(args):
    vocab = build_vocab(json_path=args.caption_path, count_thr=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str,
                        default='radcap_data.json',
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='/home/ehb/PycharmProjects/radcap_project/data/vocab.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=10,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)



