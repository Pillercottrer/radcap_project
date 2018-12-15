
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import pickle
import json
import random
import os
import numpy as np
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import torchvision.datasets as datasets
from cnn_rnn_models.cnn_rnn_xu.model_cnnrnn_attention_sgrvinod import Encoder, DecoderWithAttention
from cnn_rnn_models.cnn_rnn_xu.dataHandlerAttention import get_loader
from nltk.translate.bleu_score import corpus_bleu
from torch.optim import lr_scheduler
from data_preprocessing.build_vocab import Vocabulary



# Data parameters
data_folder = '/media/ssd/caption data'  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
data_vocab_path = './data/vocab.pkl' #path for vocabulary
root_image_path = './data/'
model_path = './models/'

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 120  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 32
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none


def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map, vocab

    """
    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    """
    # Load vocabulary wrapper
    with open(data_vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(vocab),
                                       dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    """
    #Flickr
    # Split: val, train, test
    flickrimgs = json.load(open('./all_imgcap.json', 'r'))
    random.shuffle(flickrimgs)
    len_train = int(round(0.7 * len(flickrimgs)))
    len_val = int(round(0.20 * len(flickrimgs)))

    train_imgs = flickrimgs[:len_train]
    val_imgs = flickrimgs[len_train:len_train + len_val]
    test_imgs = flickrimgs[len_train + len_val:]

    json.dump(test_imgs, open('flickr_test_data.json', 'w'))

    image_datasets = {}
    image_datasets['train'] = train_imgs
    image_datasets['val'] = val_imgs
    image_datasets['test'] = test_imgs
    """

    # Radiology
    radimgs = json.load(open('./data_preprocessing/radcap_bodypartsplit_data.json', 'r'))
    radimgs_ankle = radimgs['ankle']
    random.shuffle(radimgs_ankle)
    radimgs_fracture = []

    for jimg in radimgs_ankle:
        if int(jimg['Fracture']) > 0 and int(jimg['Implant']) < 0:
            #radimgs_fracture.append(jimg)
            if 'oförändra' not in jimg['paragraph'][0] and 'Oförändra' not in jimg['paragraph'][0]:
                radimgs_fracture.append(jimg)



    len_train = int(round(0.7 * len(radimgs_fracture)))
    len_val = int(round(0.20 * len(radimgs_fracture)))

    train_imgs = radimgs_fracture[:len_train]
    val_imgs = radimgs_fracture[len_train:len_train + len_val]
    test_imgs = radimgs_fracture[len_train + len_val:]

    json.dump(test_imgs, open('ankle_test_data_only_fracture_without_checkup.json', 'w'))

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

    #Dataloaders
    train_loader = get_loader(root_image_path, image_datasets['train'], vocab, data_transforms['train'], batch_size=batch_size, shuffle=True,
                                 num_workers=workers)

    val_loader = get_loader(root_image_path, image_datasets['val'], vocab, data_transforms['val'], batch_size=batch_size, shuffle=True,
                                 num_workers=workers)

    # Decay LR by a factor of 0.1 every 7 epochs
    if fine_tune_encoder:
        exp_lr_scheduler_encoder = lr_scheduler.StepLR(encoder_optimizer, step_size=8, gamma=0.1)
    exp_lr_scheduler_decoder = lr_scheduler.StepLR(decoder_optimizer, step_size=8, gamma=0.1)

    # Epochs
    for epoch in range(start_epoch, epochs):

        exp_lr_scheduler_decoder.step()
        if fine_tune_encoder:
            exp_lr_scheduler_encoder.step()

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 10:
            break
        """
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            #adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                #adjust_learning_rate(encoder_optimizer, 0.8)
        """

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            # Save the model checkpoints
            torch.save(decoder.state_dict(), os.path.join(model_path, 'decoder-{}-{}.ckpt'.format('best', 'bleu4')))
            torch.save(encoder.state_dict(), os.path.join(model_path, 'encoder-{}-{}.ckpt'.format('best', 'bleu4')))
            epochs_since_improvement = 0

        """
        # Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)
        """
    # Save the model checkpoints
    torch.save(decoder.state_dict(), os.path.join(model_path, 'decoder-{}-{}.ckpt'.format(epoch + 1, 'attention-final')))
    torch.save(encoder.state_dict(), os.path.join(model_path, 'encoder-{}-{}.ckpt'.format(epoch + 1, 'attention-final')))


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.
    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        #data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = np.array(caplens)
        caplens = torch.from_numpy(caplens)
        caplens = caplens.to(device)

        # Forward prop.
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder, grad_clip) #decoder or decoder_optimizer?
            if encoder_optimizer is not None:
                clip_gradient(encoder, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        #top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        #top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()


        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses))



def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    #top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # Batches
    for i, (imgs, caps, caplens) in enumerate(val_loader):

        # Move to device, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = np.array(caplens)
        caplens = torch.from_numpy(caplens)
        caplens = caplens.to(device)

        # Forward prop.
        if encoder is not None:
            imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores_copy = scores.clone()
        scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        loss = criterion(scores, targets)

        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Keep track of metrics
        losses.update(loss.item(), sum(decode_lengths))
        #top5 = accuracy(scores, targets, 5)
        #top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()
        """
        if i % print_freq == 0:
            print('Validation: [{0}/{1}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                            loss=losses, top5=top5accs))
        """
        if i % print_freq == 0:
            print('Validation: [{0}/{1}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                            loss=losses))

        # Store references (true captions), and hypothesis (prediction) for each image
        # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
        # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
        # References
        """
        allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
        for j in range(allcaps.shape[0]):
            img_caps = allcaps[j].tolist()
            img_captions = list(
                map(lambda c: [w for w in c if w not in {vocab('<start>'), vocab('<pad>')}],
                    img_caps))  # remove <start> and pads
            references.append(img_captions)
        """
        for cap in caps_sorted:
            img_cap_list = cap.tolist()
            img_cap_removed_start_and_pad = [w for w in img_cap_list if w not in {vocab('<start>'), vocab('<pad>')}]  # remove <start> and pads)
            references.append([img_cap_removed_start_and_pad])

        # Hypotheses
        _, preds = torch.max(scores_copy, dim=2)
        preds = preds.tolist()
        temp_preds = list()
        for j, p in enumerate(preds):
            temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
        preds = temp_preds
        hypotheses.extend(preds)

        assert len(references) == len(hypotheses)

    # Calculate BLEU-4 scores
    #bleu4 = corpus_bleu(references, hypotheses, emulate_multibleu=True)
    bleu4 = corpus_bleu(references, hypotheses)

    print(
        '\n * LOSS - {loss.avg:.3f}, BLEU-4 - {bleu}\n'.format(
            loss=losses,
            bleu=bleu4))

    return bleu4
"""
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
"""
def clip_gradient(model, clip):
    if clip is None:
        return
    totalnorm = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        p.grad.data = p.grad.data.clamp(-clip, clip)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = optimizer.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        #maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()