import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import json
import os
from torchvision import transforms
from data_preprocessing.build_vocab import Vocabulary
from cnn_rnn_models.cnn_rnn_vinalys.model_cnn_rnn import EncoderCNN, DecoderRNN
from PIL import Image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_image(image_paths, transform=None):
    #image = Image.open(image_path).convert('RGB')
    #image = image.resize([224, 224], Image.LANCZOS)

    #if transform is not None:
    #    image = transform(image).unsqueeze(0)

    image_tensor = torch.zeros(len(image_paths), 3, 224, 224).to(device)

    for i, path in enumerate(image_paths):
        image = Image.open(path).convert('RGB')
        if transform is not None:
            image = transform(image)
        image_tensor[i] = image.to(device)

    return image_tensor


def main(args, jimg):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    # Prepare an image
    image = load_image(jimg['file_paths'], transform)
    image_tensor = image.to(device)

    # Generate an caption from the image
    feature = encoder(image_tensor.unsqueeze(0))
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()  # (1, max_seq_length) -> (max_seq_length)

    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)

    # Print out the image and the generated caption
    print('AI-cap: ' + sentence)
    print('Reference cap: ' + jimg['paragraph'][0])
    #image = Image.open(args.image)
    #plt.imshow(np.asarray(image))
    #plt.show()
    return sentence

def generate_radiology_reports(args):
    #ankle_json_with_ai_cap = []
    jimgs = json.load(open(args.test_folder_path))
    for jimg in jimgs[:40]:
        # Encode, decode with attention and beam search
        aicap = main(args, jimg)

        #jimg['aicap'] = aicap
        #ankle_json_with_ai_cap.append(jimg)

    #json.dump(ankle_json_with_ai_cap, open('ankle_with_ai_cap.json', 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='./models/Vinalys_barafraktur_handled/encoder-Vinalys-wrist-end.ckpt',
                        help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='./models/Vinalys_barafraktur_handled/decoder-Vinalys-wrist-end.ckpt',
                        help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--test_folder_path', type=str, default='./models/Vinalys_barafraktur_handled/wrist_test_data_only_fracture_without_checkup.json', help='path for folder with testimgs')

    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')
    args = parser.parse_args()

    # Load test imgs
    generate_radiology_reports(args)