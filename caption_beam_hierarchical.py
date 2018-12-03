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
from scipy.misc import imread, imresize
from PIL import Image
from cnn_rnn_models.cnn_rnn_hierarchical.model_hierarchical import Encoder, CoAttention, MLC, SentenceLSTMDecoder, Embedding, WordLSTMDecoder
from data_preprocessing.build_vocab import Vocabulary


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_vocab_path = './data/vocab.pkl' #path for vocabulary
model_path = './models'

# Load vocabulary wrapper
with open(data_vocab_path, 'rb') as f:
    vocab = pickle.load(f)
vocab_size = len(vocab)

def main(beam_size=3):
    vocab = word_map
    vocab_size = len(vocab)

    # Read image and process
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)


