import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
import pickle
import imageio
#from scipy.misc import imread, imresize
from imageio import imread
from skimage.transform import resize
from PIL import Image
from cnn_rnn_models.cnn_rnn_hierarchical.model_hierarchical import Encoder, CoAttention, MLC, SentenceLSTMDecoder, Embedding, WordLSTMDecoder
from data_preprocessing.build_vocab import Vocabulary


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
radimgs_test = json.load(open('./ankle_test_data.json', 'r'))
data_vocab_path = './data/vocab.pkl' #path for vocabulary
model_path = './models'

# Load vocabulary wrapper
with open(data_vocab_path, 'rb') as f:
    vocab = pickle.load(f)
vocab_size = len(vocab)

#parameters
num_pixels_visual_attention = 49
encoder_dim = 512


def getImageTensor(rad_jimg):
    paths = rad_jimg['file_paths']  # radiology

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])

    # IMAGE
    image_tensor = torch.zeros(len(paths), 3, 224, 224).to(device)

    for i, path in enumerate(paths):
        #image = Image.open(path).convert('RGB')
        # Read image and process
        img = imread(path)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        img = resize(img, (224, 224))
        img = img.transpose(2, 0, 1)
        img = img / 255.
        img = torch.FloatTensor(img).to(device)
        if transform is not None:
            img = transform(img)
        image_tensor[i] = img
    return image_tensor



def main(beam_size=3):

    json_test_images = radimgs_test[7:10]

    # Load models
    encoder = Encoder()
    encoder = encoder.to(device)
    encoder.load_state_dict(torch.load('./models/encoder-best-cider-hierarchical.ckpt'))

    embedding = Embedding(vocab_size)
    embedding = embedding.to(device)
    embedding.load_state_dict(torch.load('./models/embedding-best-cider-hierarchical.ckpt'))

    mlc = MLC(num_pixels_visual_attention, encoder_dim, embedding, vocab)
    mlc = mlc.to(device)
    mlc.load_state_dict(torch.load('./models/mlc-best-cider-hierarchical.ckpt'))

    sent_lstm = SentenceLSTMDecoder(vocab_size)
    sent_lstm.to(device)
    sent_lstm.load_state_dict(torch.load('./models/sent-lstm-best-cider-hierarchical.ckpt'))

    word_lstm = WordLSTMDecoder(vocab_size, embedding)
    word_lstm.to(device)
    word_lstm.load_state_dict(torch.load('./models/word-lstm-best-cider-hierarchical.ckpt'))

    encoder.eval()
    embedding.eval()
    mlc.eval()
    sent_lstm.eval()
    word_lstm.eval()
    for i, jimg in enumerate(json_test_images):
        print('test')
        image_tensor = getImageTensor(jimg)
        image_tensor = image_tensor.unsqueeze(0)

        # Encode images
        encoder_out = encoder(image_tensor)  # resnet-output
        # Flatten image
        visual_features = encoder_out.view(1, -1,
                                           encoder_dim)  # (batch_size, num_pixels, encoder_dim)

        semantic_features, tag_scores = mlc(visual_features)

        topic_tensor, stop_tensor, alphas_visual, alphas_semantic = sent_lstm(visual_features, semantic_features)
        counter = 0
        for i, t in enumerate(stop_tensor[0]):
            if t[1].item() < 0.5:
                counter += 1
            else:
                break

        # generate sentences
        sampled_sents = word_lstm.sample_greedy(topic_tensor[0, :counter]) # correct?

        #print('Reference radiology report: "{0}"'.format(jimg['paragraph']))
        #print('Generated radiology report: "{0}"'.format(' '.join(sampled_sents)))

if __name__ == '__main__':
    main()