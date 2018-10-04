import json
import os
import string
import argparse
from random import shuffle, seed

#special
import h5py
import numpy as np
from scipy.misc import imread, imresize
unknown_token = "<unk>"
sentence_start_token = "<start>"
sentence_end_token = "<end>"

#params = {}

#params['max_length'] = 16
#params['word_count_threshhold'] = 5




def prepro_captions(imgs):
    # preprocess all the captions
    print('example processed tokens:')
    for i, img in enumerate(imgs):
        img['processed_tokens'] = []
        for j, s in enumerate(img['captions']):
            txt = str(s).lower().translate(str.maketrans("", "", string.punctuation)).strip().split()
            img['processed_tokens'].append(txt)
            if i < 10 and j == 0:
                print(txt)


def build_vocab(imgs, params):
    count_thr = params['word_count_threshold']

    # count up the number of words
    counts = {}
    for img in imgs:
        for txt in img['processed_tokens']:
            for w in txt:
                counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    print('top words and their counts:')
    print('\n'.join(map(str, cw[:20])))

    # print some stats
    total_words = sum(counts.values())
    print('total words:', total_words)
    bad_words = [w for w, n in counts.items() if n <= count_thr]
    vocab = [w for w, n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % (len(vocab),))
    print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count * 100.0 / total_words))

    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for img in imgs:
        for txt in img['processed_tokens']:
            nw = len(txt)
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    print('max length sentence in raw data: ', max_len)
    print('sentence length distribution (count, number of words):')
    sum_len = sum(sent_lengths.values())
    for i in range(max_len + 1):
        print('%2d: %10d   %f%%' % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0) * 100.0 / sum_len))

    #inserting start and end tokens to vocab
    print('inserting start and end tokens')
    vocab.append('<start>')
    vocab.append('<end>')
    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')
        vocab.append('UNK')


    for img in imgs:
        img['final_captions'] = []
        for txt in img['processed_tokens']:
            caption = []
            caption.append('<start>')
            caption.extend([w if counts.get(w, 0) > count_thr else 'UNK' for w in txt])
            caption.append('<end>')
            img['final_captions'].append(caption)

    return vocab


def assign_splits(imgs, params):
  num_val = params['num_val']
  num_test = params['num_test']

  for i,img in enumerate(imgs):
      if i < num_val:
        img['split'] = 'val'
      elif i < num_val + num_test:
        img['split'] = 'test'
      else:
        img['split'] = 'train'

  print('assigned %d to val, %d to test.' % (num_val, num_test))


def encode_captions(imgs, params, wtoi):
    """
    encode all captions into one large array, which will be 1-indexed.
    also produces label_start_ix and label_end_ix which store 1-indexed
    and inclusive (Lua-style) pointers to the first and last caption for
    each image in the dataset.
    """

    """
    kreygasm
    kreygasm
    kreygasm
    kreygasm
    """
    max_length = params['max_length']
    N = len(imgs)
    M = sum(len(img['final_captions']) for img in imgs)  # total number of captions

    label_arrays = []
    label_start_ix = np.zeros(N, dtype='uint32')  # note: these will be one-indexed
    label_end_ix = np.zeros(N, dtype='uint32')
    label_length = np.zeros(M, dtype='uint32')
    caption_counter = 0
    counter = 1
    for i, img in enumerate(imgs):
        n = len(img['final_captions'])
        assert n > 0, 'error: some image has no captions'

        Li = np.zeros((n, max_length), dtype='uint32')
        for j, s in enumerate(img['final_captions']):
            label_length[caption_counter] = min(max_length, len(s))  # record the length of this sequence
            caption_counter += 1
            for k, w in enumerate(s):
                if k < max_length:
                    Li[j, k] = wtoi[w]

        # note: word indices are 1-indexed, and captions are padded with zeros
        label_arrays.append(Li)
        label_start_ix[i] = counter
        label_end_ix[i] = counter + n - 1

        counter += n

    L = np.concatenate(label_arrays, axis=0)  # put all the labels together
    assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
    assert np.all(label_length > 0), 'error: some caption had no words?'

    print('encoded captions to array of size ', repr(L.shape))
    return L, label_start_ix, label_end_ix, label_length


#with open('./data/data_file_imgcap_train.json') as json_file:
    #data = json.load(json_file)

#prepro_captions(data[1:10])


def main(params):
    imgs = json.load(open(params['input_json'], 'r'))
    seed(123)  # make reproducible
    shuffle(imgs)  # shuffle the order

    # tokenization and preprocessing
    prepro_captions(imgs)

    # create the vocab
    vocab = build_vocab(imgs, params)
    itow = {i + 1: w for i, w in enumerate(vocab)}  # a 1-indexed vocab translation table
    wtoi = {w: i + 1 for i, w in enumerate(vocab)}  # inverse table

    # assign the splits
    assign_splits(imgs, params)

    # encode captions in large arrays, ready to ship to hdf5 file
    L, label_start_ix, label_end_ix, label_length = encode_captions(imgs, params, wtoi)

    # create output h5 file
    N = len(imgs)
    f = h5py.File(params['output_h5'], "w")
    f.create_dataset("labels", dtype='uint32', data=L)
    f.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
    f.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
    f.create_dataset("label_length", dtype='uint32', data=label_length)
    dset = f.create_dataset("images", (N, 3, 224, 224), dtype='uint8')  # space for resized images
    for i, img in enumerate(imgs):
        # load the image
        I = imread(os.path.join(params['images_root'], img['filepath']))
        try:
            Ir = imresize(I, (224, 224))
        except:
            print('failed resizing image %s - see http://git.io/vBIE0' % (img['file_path'],))
            raise
        # handle grayscale input images
        if len(Ir.shape) == 2:
            Ir = Ir[:, :, np.newaxis]
            Ir = np.concatenate((Ir, Ir, Ir), axis=2)
        # and swap order of axes from (256,256,3) to (3,256,256)
        Ir = Ir.transpose(2, 0, 1)
        # write to h5
        dset[i] = Ir
        if i % 1000 == 0:
            print('processing %d/%d (%.2f%% done)' % (i, N, i * 100.0 / N))
    f.close()
    print('wrote ', params['output_h5'])

    # create output json file
    out = {}
    out['ix_to_word'] = itow  # encode the (1-indexed) vocab
    out['images'] = []
    for i, img in enumerate(imgs):

        jimg = {}
        jimg['split'] = img['split']
        if 'filepath' in img: jimg['filepath'] = img['filepath']  # copy it over, might need
        if 'id' in img: jimg['id'] = img['id']  # copy over & mantain an id, if present (e.g. coco ids, useful)

        out['images'].append(jimg)

    json.dump(out, open(params['output_json'], 'w'))
    print('wrote ', params['output_json'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', required=True, help='input json file to process into hdf5')
    parser.add_argument('--num_val', required=True, type=int,
                        help='number of images to assign to validation data (for CV etc)')
    parser.add_argument('--output_json', default='data.json', help='output json file')
    parser.add_argument('--output_h5', default='data.h5', help='output h5 file')

    # options
    parser.add_argument('--max_length', default=16, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--images_root', default='',
                        help='root location in which images are stored, to be prepended to file_path in input json')
    parser.add_argument('--word_count_threshold', default=5, type=int,
                        help='only words that occur more than this number of times will be put in vocab')
    parser.add_argument('--num_test', default=0, type=int,
                        help='number of test images (to withold until very very end)')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)
