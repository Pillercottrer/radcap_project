import pickle
import torchvision.transforms as transforms
import json
import random
from cnn_rnn_models.cnn_rnn_hierarchical.dataHandlerHierarchical import get_loader
from cnn_rnn_models.cnn_rnn_hierarchical.model_hierarchical import Encoder, CoAttention, MLC, SentenceLSTMDecoder, Embedding, WordLSTMDecoder
from data_preprocessing.build_vocab import Vocabulary

data_vocab_path = './data/vocab.pkl' #path for vocabulary

#Training parameters
batch_size = 32
workers = 1



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


    for i, (images, paragraph_tensor, sentence_lengths_tensor, num_sents) in enumerate(train_loader):
        print('test')





if __name__ == '__main__':
    main()