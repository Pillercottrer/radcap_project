import pickle
import torchvision.transforms as transforms
import json
import random
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from cnn_rnn_models.cnn_rnn_hierarchical.dataHandlerHierarchical import get_loader
from cnn_rnn_models.cnn_rnn_hierarchical.model_hierarchical import Encoder, CoAttention, MLC, SentenceLSTMDecoder, Embedding, WordLSTMDecoder
from data_preprocessing.build_vocab import Vocabulary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_vocab_path = './data/vocab.pkl' #path for vocabulary

#Training parameters
batch_size = 32
workers = 1
encoder_lr = 1e-5
mlc_lr = 1e-5
sent_lstm_lr = 5e-4
word_lstm_lr = 5e-4
fine_tune_encoder = True




def main():
    print('test')

    # Load vocabulary wrapper
    with open(data_vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Radiology
    radimgs = json.load(open('./data_preprocessing/radcap_bodypartsplit_data.json', 'r'))
    radimgs_ankle = radimgs['ankle']
    random.shuffle(radimgs_ankle)
    len_train = int(round(0.7 * len(radimgs_ankle)))
    len_val = int(round(0.20 * len(radimgs_ankle)))

    train_imgs = radimgs_ankle[:len_train]
    val_imgs = radimgs_ankle[len_train:len_train + len_val]
    test_imgs = radimgs_ankle[len_train + len_val:]

    #json.dump(test_imgs, open('ankle_test_data.json', 'w'))

    image_datasets = {}
    image_datasets['train'] = train_imgs
    image_datasets['val'] = val_imgs
    image_datasets['test'] = test_imgs

    for jimg in train_imgs:
        jimg['split'] = 'train'
    for jimg in val_imgs:
        jimg['split'] = 'val'

    # Image preprocessing, normalization for the pretrained resnet
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    train_loader = get_loader(image_datasets['train'], vocab, data_transforms['train'],
                              batch_size=batch_size, shuffle=True,
                              num_workers=workers)






    #Parameters
    vocab_size = len(vocab)
    num_pixels_visual_attention = 49
    encoder_dim = 512

    #Load models
    encoder = Encoder()
    encoder = encoder.to(device)

    embedding = Embedding(vocab_size)
    embedding = embedding.to(device)

    mlc = MLC(num_pixels_visual_attention, encoder_dim, embedding, vocab)
    mlc = mlc.to(device)

    sent_lstm = SentenceLSTMDecoder(vocab_size)
    sent_lstm.to(device)

    word_lstm = WordLSTMDecoder(vocab_size, embedding)
    word_lstm.to(device)
    #Optimizers
    mlc_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, mlc.parameters()),
                                     lr=mlc_lr)

    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                         lr=encoder_lr) if fine_tune_encoder else None

    sent_lstm_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, sent_lstm.parameters()),
                                     lr=sent_lstm_lr)

    word_lstm_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, word_lstm.parameters()),
                                     lr=mlc_lr)

    # Loss function
    criterion = torch.nn.CrossEntropyLoss().to(device)


    for i, (images, target_paragraphs, tags, paragraph_sent_lengths, num_sents, max_length) in enumerate(train_loader):
        print('test')
        images = images.to(device)
        num_sents = torch.Tensor(num_sents).long().to(device)
        target_paragraphs = target_paragraphs.to(device)
        paragraph_sent_lengths = paragraph_sent_lengths.to(device)

        #Encode images
        encoder_out = encoder(images)  #resnet-output
        # Flatten image
        visual_features = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)

        semantic_features = mlc(visual_features)

        topic_tensor, stop_tensor = sent_lstm(visual_features, semantic_features)

        predictions, sorted_paragraphs, sorted_sent_lengths, sort_ind_num_sent = word_lstm(topic_tensor, num_sents, target_paragraphs, paragraph_sent_lengths, max(max_length))


        for j in range(sorted_paragraphs.size(1)):  #Loop over all sentences and compute comulative loss over each sentence.
            print('breakpoint')
            #scores, _ = pack_padded_sequence(predictions[:, j], sorted_sent_lengths[:, j], batch_first=True)
            #target, _ = pack_padded_sequence(sorted_paragraphs[:, j], sorted_sent_lengths[:, j], batch_first=True)

        print('breakpoint')














if __name__ == '__main__':
    main()