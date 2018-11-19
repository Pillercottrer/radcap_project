import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import json
import pickle
import random
import copy
import time
from dataHandler import get_loader
from build_vocab import Vocabulary
from model_cnn_rnn import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from torch.optim import lr_scheduler


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Data augmentation and normalization for training
    # Just normalization for validation

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

    # Split: val, train, test


    #COCO
    """
    cocoimgs = json.load(open('./coco_raw.json', 'r'))
    random.shuffle(cocoimgs)
    len_train = int(round(0.7 * len(cocoimgs)))
    len_val = int(round(0.20 * len(cocoimgs)))

    train_imgs = cocoimgs[:len_train]
    val_imgs = cocoimgs[len_train:len_train + len_val]
    test_imgs = cocoimgs[len_train + len_val:]


    image_datasets = {}
    image_datasets['train'] = train_imgs
    image_datasets['val'] = val_imgs
    image_datasets['test'] = test_imgs
    
    """
    #Flicker
    """
    flicker_trainimgs = json.load(open('./train_imgcap.json', 'r'))
    flicker_testimgs = json.load(open('./test_imgcap.json', 'r'))
    len_train = len(flicker_trainimgs)
    len_val = len(flicker_testimgs)

    image_datasets = {}
    image_datasets['train'] = flicker_trainimgs
    image_datasets['val'] = flicker_testimgs
    #image_datasets['test'] = test_imgs
    """
    #Radiology
    radimgs = json.load(open('./radcap_bodypartsplit_data.json', 'r'))
    radimgs_ankle = radimgs['ankle']
    random.shuffle(radimgs_ankle)
    len_train = int(round(0.7 * len(radimgs_ankle)))
    len_val = int(round(0.20 * len(radimgs_ankle)))

    train_imgs = radimgs_ankle[:len_train]
    val_imgs = radimgs_ankle[len_train:len_train + len_val]
    test_imgs = radimgs_ankle[len_train + len_val:]

    json.dump(test_imgs, open('ankle_test_data.json', 'w'))

    image_datasets = {}
    image_datasets['train'] = train_imgs
    image_datasets['val'] = val_imgs
    image_datasets['test'] = test_imgs

    #Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)


    #with open('data.json') as json_file:
    #    json_data = json.load(json_file)
    #vocab = json_data['ix_to_word']



    # Build data loader
    dataloaders = {x: get_loader(args.image_dir, image_datasets[x], vocab, data_transforms['train'], args.batch_size, shuffle=True,
                                 num_workers=args.num_workers)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    #class_names = image_datasets['train'].classes

    # Build the models
    encoder = EncoderCNN(args.embed_size).to(device)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


    # Train the models
    since = time.time()

    best_model_encoder_wts = copy.deepcopy(encoder.state_dict())
    best_model_decoder_wts = copy.deepcopy(decoder.state_dict())
    lowest_loss = 100000


    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            total_step = len(dataloaders['train'])
            if phase == 'train':
                exp_lr_scheduler.step()
                encoder.train()  # Set model to training mode
            else:
                encoder.eval()  # Set model to evaluate mode

            running_loss = 0.0

            for i, (images, captions, lengths) in enumerate(dataloaders[phase]):
                # Set mini-batch dataset
                images = images.to(device)
                captions = captions.to(device)
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Forward, backward and optimize
                    features = encoder(images)
                    outputs = decoder(features, captions, lengths)
                    loss = criterion(outputs, targets)
                    decoder.zero_grad()
                    encoder.zero_grad()
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                """
                # Print log info
                if i % args.log_step == 0:
                    print('i: %d' % i)
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                        .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))
                    
                """
            # statistics
            running_loss += loss.item()*images.size(0)
            epoch_loss = running_loss / dataset_sizes[phase]
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            # deep copy the model
            if phase == 'val' and epoch_loss < lowest_loss:
                lowest_loss = epoch_loss
                best_model_encoder_wts = copy.deepcopy(encoder.state_dict())
                best_model_decoder_wts = copy.deepcopy(decoder.state_dict())
        print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Lowest loss: {:4f}'.format(lowest_loss))

        """
        for i, (images, captions, lengths) in enumerate(dataloaders['train']):

            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % args.log_step == 0:
                print('i: %d' % i)
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, args.num_epochs, i, total_step, loss.item(), np.exp(loss.item())))

                # Save the model checkpoints
            if (i + 1) % args.save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    args.model_path, 'decoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))
                torch.save(encoder.state_dict(), os.path.join(
                    args.model_path, 'encoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))
        """


    # Save the model checkpoints
    torch.save(best_model_decoder_wts, os.path.join(
        args.model_path, 'decoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))
    torch.save(best_model_encoder_wts, os.path.join(
        args.model_path, 'encoder-{}-{}.ckpt'.format(epoch + 1, i + 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/', help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='./data/', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='train_imgcap.json',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')

    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
