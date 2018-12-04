import pickle
import torchvision.transforms as transforms
import json
import random
import torch
import os
from torch.nn.utils.rnn import pack_padded_sequence
from cnn_rnn_models.cnn_rnn_hierarchical.dataHandlerHierarchical import get_loader
from cnn_rnn_models.cnn_rnn_hierarchical.model_hierarchical import Encoder, CoAttention, MLC, SentenceLSTMDecoder, Embedding, WordLSTMDecoder
from caption_eval.cider.cider import Cider
from data_preprocessing.build_vocab import Vocabulary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_vocab_path = './data/vocab.pkl' #path for vocabulary
model_path = './models'

#Training parameters
batch_size = 32
workers = 0
encoder_lr = 1e-5
mlc_lr = 1e-5
sent_lstm_lr = 5e-4
word_lstm_lr = 5e-4
fine_tune_encoder = True
num_epochs = 30

# Load vocabulary wrapper
with open(data_vocab_path, 'rb') as f:
    vocab = pickle.load(f)
vocab_size = len(vocab)
#Parameters
max_num_sents = 8
num_pixels_visual_attention = 49
encoder_dim = 512
tags = ['normal', 'fracture', 'implant', 'tumor', 'osteoarthritis']


def main():

    # Radiology
    radimgs = json.load(open('./data_preprocessing/radcap_bodypartsplit_data.json', 'r'))
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
    val_loader = get_loader(image_datasets['val'], vocab, data_transforms['val'],
                              batch_size=batch_size, shuffle=True,
                              num_workers=workers)

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
    mlc_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, mlc.parameters()), lr=mlc_lr)

    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                         lr=encoder_lr) if fine_tune_encoder else None

    sent_lstm_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, sent_lstm.parameters()),
                                     lr=sent_lstm_lr)

    word_lstm_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, word_lstm.parameters()),
                                     lr=mlc_lr)




    criterion = torch.nn.CrossEntropyLoss().to(device)

    criterion_sentence = torch.nn.MSELoss().to(device)

    best_CIDEr_score = 0
    for epoch in range(num_epochs):
        print('Epoch:{0}'.format(epoch))
        train(train_loader=train_loader, encoder=encoder, mlc=mlc, sent_lstm=sent_lstm, word_lstm=word_lstm, criterion=criterion, encoder_optimizer=encoder_optimizer, mlc_optimizer=mlc_optimizer, word_lstm_optimizer=word_lstm_optimizer, sent_lstm_optimizer=sent_lstm_optimizer, epoch=epoch)

        CIDEr_score = validate(val_loader=val_loader, encoder=encoder, mlc=mlc, sent_lstm=sent_lstm, word_lstm=word_lstm, criterion=criterion)
        if CIDEr_score > best_CIDEr_score:
            best_CIDEr_score = CIDEr_score
            torch.save(encoder.state_dict(), os.path.join(model_path, 'encoder-{}.ckpt'.format('best-cider-hierarchical')))
            torch.save(embedding.state_dict(),
                       os.path.join(model_path, 'embedding-{}.ckpt'.format('best-cider-hierarchical')))
            torch.save(mlc.state_dict(),
                       os.path.join(model_path, 'mlc-{}.ckpt'.format('best-cider-hierarchical')))
            torch.save(sent_lstm.state_dict(),
                       os.path.join(model_path, 'sent-lstm-{}.ckpt'.format('best-cider-hierarchical')))
            torch.save(word_lstm.state_dict(),
                       os.path.join(model_path, 'word-lstm-{}.ckpt'.format('best-cider-hierarchical')))



    torch.save(encoder.state_dict(), os.path.join(model_path, 'encoder-{}.ckpt'.format('final-hierarchical')))
    torch.save(embedding.state_dict(),
               os.path.join(model_path, 'embedding-{}.ckpt'.format('final-hierarchical')))
    torch.save(mlc.state_dict(),
               os.path.join(model_path, 'mlc-{}.ckpt'.format('final-hierarchical')))
    torch.save(sent_lstm.state_dict(),
               os.path.join(model_path, 'sent-lstm-{}.ckpt'.format('final-hierarchical')))
    torch.save(word_lstm.state_dict(),
               os.path.join(model_path, 'word-lstm-{}.ckpt'.format('final-hierarchical')))

    """
    lowest_loss = 10000
    for i, (images, target_list_paragraph, batch_sizes, tags, tags_yn, paragraph_sent_lengths, num_sents, max_length) in enumerate(train_loader):

        for batch_index, target_paragraphs in enumerate(target_list_paragraph):
            print('test')
            index_range_start = sum(batch_sizes[:batch_index])
            index_range_end = index_range_start + batch_sizes[batch_index]

            mini_batch_size = batch_sizes[batch_index]


            images_minibatch = images[index_range_start:index_range_end].to(device)
            num_sents_minibatch = torch.Tensor(num_sents[index_range_start:index_range_end]).long().to(device)
            target_paragraphs_minibatch = target_paragraphs.to(device)
            paragraph_sent_lengths_minibatch = paragraph_sent_lengths[index_range_start:index_range_end].to(device)
            tags_minibatch = tags_yn[index_range_start:index_range_end].to(device)
            max_length_minibatch = max_length[index_range_start:index_range_end]

            #Encode images
            encoder_out = encoder(images_minibatch)  #resnet-output
            # Flatten image
            visual_features = encoder_out.view(mini_batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)

            semantic_features, tag_scores = mlc(visual_features)

            topic_tensor, stop_tensor, alphas_visual, alphas_semantic = sent_lstm(visual_features, semantic_features)




            #generate sentences
            predictions, sorted_paragraphs, sorted_sent_lengths, sort_ind_num_sent = word_lstm(topic_tensor, num_sents_minibatch, target_paragraphs_minibatch, paragraph_sent_lengths_minibatch, max(max_length))
            
            # Loop over all sentences
            #for i, p in enumerate(sorted_paragraphs):
            #    print('breakpoint')
            #    #sorted_p_ind = torch.sort(sorted_sent_lengths[i], descending=True)
            #    for k, s in enumerate(p):
            #        print('breakpoint')
            

            #Calculate tags classification loss

            scores_tags, _ = pack_padded_sequence(tag_scores, [5]*mini_batch_size,
                                             batch_first=True)

            targets_tags, _ = pack_padded_sequence(tags_minibatch, [5]*mini_batch_size,
                                                  batch_first=True)

            tag_loss = criterion(scores_tags, targets_tags)

            # Calculate sentence lstm loss

            if num_sents_minibatch[0] < 8:
                target_sentence_lstm = torch.zeros(mini_batch_size, (num_sents_minibatch[0] + 1)).long().to(device)
                target_sentence_lstm[:, -1] = 1

                scores_sentence_lstm = stop_tensor[:, :(num_sents_minibatch[0] + 1)]

                lens = [num_sents_minibatch[0] + 1] * mini_batch_size
            else:
                target_sentence_lstm = torch.zeros(mini_batch_size, 8).long().to(device)

                scores_sentence_lstm = stop_tensor[:, :8]

                lens = [8] * mini_batch_size



            scores_sentence_lstm, _ = pack_padded_sequence(scores_sentence_lstm[:len(lens)], lens,
                                             batch_first=True)
            target_sentence_lstm, _ = pack_padded_sequence(target_sentence_lstm[:len(lens)], lens,
                                              batch_first=True)


            sentence_lstm_loss = criterion(scores_sentence_lstm, target_sentence_lstm)

            total_losses = []
            #Loop over all sentences and compute comulative loss over each sentence
            for j in range(sorted_paragraphs.size(1)):
                print('breakpoint')
                if mini_batch_size > 1:
                    sorted_sent_length_batch, sort_ind = torch.sort(sorted_sent_lengths[:, j], descending=True)

                    sorted_sent_length_batch = sorted_sent_length_batch[sorted_sent_length_batch.nonzero().squeeze()]

                    sentence_batch_unsorted = sorted_paragraphs[:, j]
                    sentence_batch_sorted = sentence_batch_unsorted[sort_ind]
                    targets = sentence_batch_sorted[:, 1:]

                    predictions_batch_unsorted = predictions[:, j]
                    predictions_batch_sorted = predictions_batch_unsorted[sort_ind]
                    scores = predictions_batch_sorted

                    scores, _ = pack_padded_sequence(scores[:len(sorted_sent_length_batch)], sorted_sent_length_batch,
                                                     batch_first=True)
                    targets, _ = pack_padded_sequence(targets[:len(sorted_sent_length_batch)], sorted_sent_length_batch,
                                                      batch_first=True)

                else:
                    sorted_sent_length_batch = sorted_sent_lengths[:, j]
                    sentence_batch_sorted = sorted_paragraphs[:, j]
                    targets = sentence_batch_sorted[:, 1:]

                    predictions_batch_sorted = predictions[:, j]
                    scores = predictions_batch_sorted

                    scores, _ = pack_padded_sequence(scores[:len(sorted_sent_length_batch)], sorted_sent_length_batch,
                                                     batch_first=True)
                    targets, _ = pack_padded_sequence(targets[:len(sorted_sent_length_batch)], sorted_sent_length_batch,
                                                      batch_first=True)




                #a = sum(sorted_sent_length_batch)
                print('breakpoint')




                # Calculate loss
                loss_word_lstm = criterion(scores, targets)
                total_losses.append(loss_word_lstm)

                print('break')
            total_losses.append(tag_loss)
            total_losses.append(sentence_lstm_loss)

            total_losses = sum(total_losses)

            # Add doubly stochastic attention regularization
            total_losses += (1. - alphas_visual[:, :num_sents_minibatch[0]].sum(dim=1) ** 2).mean()
            total_losses += (1. - alphas_semantic[:, :num_sents_minibatch[0]].sum(dim=1) ** 2).mean()



            encoder_optimizer.zero_grad()
            mlc_optimizer.zero_grad()
            sent_lstm_optimizer.zero_grad()
            word_lstm_optimizer.zero_grad()

            #backprop
            total_losses.backward()

            encoder_optimizer.step()
            mlc_optimizer.step()
            sent_lstm_optimizer.step()
            word_lstm_optimizer.step()


            print('breakpoint')
    """


def train(train_loader, encoder, mlc, sent_lstm, word_lstm, criterion, encoder_optimizer, mlc_optimizer, word_lstm_optimizer, sent_lstm_optimizer, epoch):

    encoder.train()
    mlc.train()
    sent_lstm.train()
    word_lstm.train()

    lowest_loss = 10000
    for i, (images, target_list_paragraph, batch_sizes, tags, tags_yn, paragraph_sent_lengths, num_sents,
            max_length) in enumerate(train_loader):
        fifty_batch_loss = 0
        for batch_index, target_paragraphs in enumerate(target_list_paragraph):
            index_range_start = sum(batch_sizes[:batch_index])
            index_range_end = index_range_start + batch_sizes[batch_index]

            mini_batch_size = batch_sizes[batch_index]

            images_minibatch = images[index_range_start:index_range_end].to(device)
            num_sents_minibatch = torch.Tensor(num_sents[index_range_start:index_range_end]).long().to(device)
            target_paragraphs_minibatch = target_paragraphs.to(device)
            paragraph_sent_lengths_minibatch = paragraph_sent_lengths[index_range_start:index_range_end].to(device)
            tags_minibatch = tags_yn[index_range_start:index_range_end].to(device)
            max_length_minibatch = max_length[index_range_start:index_range_end]

            # Encode images
            encoder_out = encoder(images_minibatch)  # resnet-output
            # Flatten image
            visual_features = encoder_out.view(mini_batch_size, -1,
                                               encoder_dim)  # (batch_size, num_pixels, encoder_dim)

            semantic_features, tag_scores = mlc(visual_features)

            topic_tensor, stop_tensor, alphas_visual, alphas_semantic = sent_lstm(visual_features, semantic_features)

            # generate sentences
            predictions, sorted_paragraphs, sorted_sent_lengths, sort_ind_num_sent = word_lstm(topic_tensor,
                                                                                               num_sents_minibatch,
                                                                                               target_paragraphs_minibatch,
                                                                                               paragraph_sent_lengths_minibatch,
                                                                                               max(max_length))
            """
            # Loop over all sentences
            for i, p in enumerate(sorted_paragraphs):
                print('breakpoint')
                #sorted_p_ind = torch.sort(sorted_sent_lengths[i], descending=True)
                for k, s in enumerate(p):
                    print('breakpoint')
            """

            # Calculate tags classification loss

            scores_tags, _ = pack_padded_sequence(tag_scores, [5] * mini_batch_size,
                                                  batch_first=True)

            targets_tags, _ = pack_padded_sequence(tags_minibatch, [5] * mini_batch_size,
                                                   batch_first=True)

            tag_loss = criterion(scores_tags, targets_tags)

            # Calculate sentence lstm loss

            if num_sents_minibatch[0] < 8:
                target_sentence_lstm = torch.zeros(mini_batch_size, (num_sents_minibatch[0] + 1)).long().to(device)
                target_sentence_lstm[:, -1] = 1

                scores_sentence_lstm = stop_tensor[:, :(num_sents_minibatch[0] + 1)]

                lens = [num_sents_minibatch[0] + 1] * mini_batch_size
            else:
                target_sentence_lstm = torch.zeros(mini_batch_size, 8).long().to(device)

                scores_sentence_lstm = stop_tensor[:, :8]

                lens = [8] * mini_batch_size

            scores_sentence_lstm, _ = pack_padded_sequence(scores_sentence_lstm[:len(lens)], lens,
                                                           batch_first=True)
            target_sentence_lstm, _ = pack_padded_sequence(target_sentence_lstm[:len(lens)], lens,
                                                           batch_first=True)

            sentence_lstm_loss = criterion(scores_sentence_lstm, target_sentence_lstm)

            total_losses = []
            # Loop over all sentences and compute comulative loss over each sentence
            num_sents_loop_size = sorted_paragraphs.size(1) if sorted_paragraphs.size(1) < 8 else 8
            for j in range(num_sents_loop_size):
                if mini_batch_size > 1:
                    sorted_sent_length_batch, sort_ind = torch.sort(sorted_sent_lengths[:, j], descending=True)

                    sorted_sent_length_batch = sorted_sent_length_batch[sorted_sent_length_batch.nonzero().squeeze()]

                    sentence_batch_unsorted = sorted_paragraphs[:, j]
                    sentence_batch_sorted = sentence_batch_unsorted[sort_ind]
                    targets = sentence_batch_sorted[:, 1:]

                    predictions_batch_unsorted = predictions[:, j]
                    predictions_batch_sorted = predictions_batch_unsorted[sort_ind]
                    scores = predictions_batch_sorted

                    scores, _ = pack_padded_sequence(scores[:len(sorted_sent_length_batch)], sorted_sent_length_batch,
                                                     batch_first=True)
                    targets, _ = pack_padded_sequence(targets[:len(sorted_sent_length_batch)], sorted_sent_length_batch,
                                                      batch_first=True)

                else:
                    sorted_sent_length_batch = sorted_sent_lengths[:, j]
                    sentence_batch_sorted = sorted_paragraphs[:, j]
                    targets = sentence_batch_sorted[:, 1:]

                    predictions_batch_sorted = predictions[:, j]
                    scores = predictions_batch_sorted

                    scores, _ = pack_padded_sequence(scores[:len(sorted_sent_length_batch)], sorted_sent_length_batch,
                                                     batch_first=True)
                    targets, _ = pack_padded_sequence(targets[:len(sorted_sent_length_batch)], sorted_sent_length_batch,
                                                      batch_first=True)

                # Calculate loss
                loss_word_lstm = criterion(scores, targets)
                total_losses.append(loss_word_lstm)


            total_losses.append(tag_loss)
            total_losses.append(sentence_lstm_loss)

            total_losses = sum(total_losses)

            # Add doubly stochastic attention regularization
            total_losses += (1. - alphas_visual[:, :num_sents_minibatch[0]].sum(dim=1) ** 2).mean()
            total_losses += (1. - alphas_semantic[:, :num_sents_minibatch[0]].sum(dim=1) ** 2).mean()

            encoder_optimizer.zero_grad()
            mlc_optimizer.zero_grad()
            sent_lstm_optimizer.zero_grad()
            word_lstm_optimizer.zero_grad()

            # backprop
            total_losses.backward()

            encoder_optimizer.step()
            mlc_optimizer.step()
            sent_lstm_optimizer.step()
            word_lstm_optimizer.step()

            fifty_batch_loss += total_losses.item()
            if fifty_batch_loss < lowest_loss:
                lowest_loss = fifty_batch_loss
        if i % 50 is 0:
            print('Train batch loss for batch {0}: {1}'.format(i, fifty_batch_loss))

def validate(val_loader, encoder, mlc, sent_lstm, word_lstm, criterion):

    cider_score_function = Cider()
    encoder.eval()
    mlc.eval()
    sent_lstm.eval()
    word_lstm.eval()

    gts = {}
    res = {}
    references = list()  # references (true captions) for calculating CIDEr score
    hypothesis = list()  # hypotheses (predictions)
    lowest_loss = 10000
    for i, (images, target_list_paragraph, batch_sizes, tags, tags_yn, paragraph_sent_lengths, num_sents,
            max_length) in enumerate(val_loader):
        fifty_batch_loss = 0
        for batch_index, target_paragraphs in enumerate(target_list_paragraph):
            index_range_start = sum(batch_sizes[:batch_index])
            index_range_end = index_range_start + batch_sizes[batch_index]

            mini_batch_size = batch_sizes[batch_index]

            images_minibatch = images[index_range_start:index_range_end].to(device)
            num_sents_minibatch = torch.Tensor(num_sents[index_range_start:index_range_end]).long().to(device)
            target_paragraphs_minibatch = target_paragraphs.to(device)
            paragraph_sent_lengths_minibatch = paragraph_sent_lengths[index_range_start:index_range_end].to(device)
            tags_minibatch = tags_yn[index_range_start:index_range_end].to(device)
            max_length_minibatch = max_length[index_range_start:index_range_end]

            # Encode images
            encoder_out = encoder(images_minibatch)  # resnet-output
            # Flatten image
            visual_features = encoder_out.view(mini_batch_size, -1,
                                               encoder_dim)  # (batch_size, num_pixels, encoder_dim)

            semantic_features, tag_scores = mlc(visual_features)

            topic_tensor, stop_tensor, alphas_visual, alphas_semantic = sent_lstm(visual_features, semantic_features)

            # generate sentences
            predictions, sorted_paragraphs, sorted_sent_lengths, sort_ind_num_sent = word_lstm(topic_tensor,
                                                                                               num_sents_minibatch,
                                                                                               target_paragraphs_minibatch,
                                                                                               paragraph_sent_lengths_minibatch,
                                                                                               max(max_length))

            """
            # Loop over all sentences
            for i, p in enumerate(sorted_paragraphs):
                print('breakpoint')
                #sorted_p_ind = torch.sort(sorted_sent_lengths[i], descending=True)
                for k, s in enumerate(p):
                    print('breakpoint')
            """

            # Calculate tags classification loss

            scores_tags, _ = pack_padded_sequence(tag_scores, [5] * mini_batch_size,
                                                  batch_first=True)

            targets_tags, _ = pack_padded_sequence(tags_minibatch, [5] * mini_batch_size,
                                                   batch_first=True)

            tag_loss = criterion(scores_tags, targets_tags)

            # Calculate sentence lstm loss
            if num_sents_minibatch[0] < 8:
                target_sentence_lstm = torch.zeros(mini_batch_size, (num_sents_minibatch[0] + 1)).long().to(device)
                target_sentence_lstm[:, -1] = 1

                scores_sentence_lstm = stop_tensor[:, :(num_sents_minibatch[0] + 1)]

                lens = [num_sents_minibatch[0] + 1] * mini_batch_size
            else:
                target_sentence_lstm = torch.zeros(mini_batch_size, 8).long().to(device)

                scores_sentence_lstm = stop_tensor[:, :8]

                lens = [8] * mini_batch_size

            scores_sentence_lstm, _ = pack_padded_sequence(scores_sentence_lstm[:len(lens)], lens,
                                                           batch_first=True)
            target_sentence_lstm, _ = pack_padded_sequence(target_sentence_lstm[:len(lens)], lens,
                                                           batch_first=True)

            sentence_lstm_loss = criterion(scores_sentence_lstm, target_sentence_lstm)

            total_losses = []

            paragraph_reference = []
            score_list = []
            # Loop over all sentences and compute comulative loss over each sentence
            num_sents_loop_size = sorted_paragraphs.size(1) if sorted_paragraphs.size(1) < 8 else 8
            for j in range(num_sents_loop_size):
                scores_copy = torch.Tensor()
                if mini_batch_size > 1:
                    sorted_sent_length_batch, sort_ind = torch.sort(sorted_sent_lengths[:, j], descending=True)

                    sorted_sent_length_batch = sorted_sent_length_batch[sorted_sent_length_batch.nonzero().squeeze()]

                    sentence_batch_unsorted = sorted_paragraphs[:, j]
                    sentence_batch_sorted = sentence_batch_unsorted[sort_ind]
                    targets = sentence_batch_sorted[:, 1:]
                    # add to references for CIDEr scoring
                    for ind in range(targets.size(0)):
                        if j is 0:
                            paragraph_reference.append(targets[ind, :sorted_sent_length_batch[ind]])
                        else:
                            paragraph_reference[ind] = torch.cat((paragraph_reference[ind], targets[ind, :sorted_sent_length_batch[ind]]), dim=0)
                        #references.add(targets[ind, :sorted_sent_length_batch[ind]])

                    predictions_batch_unsorted = predictions[:, j]
                    predictions_batch_sorted = predictions_batch_unsorted[sort_ind]
                    scores = predictions_batch_sorted

                    # For CIDEr scoring
                    _, preds = torch.max(scores, dim=2)
                    for ind in range(preds.size(0)):
                        # temp_preds.append(preds[ind][:sorted_sent_length_batch[ind]])  # remove pads
                        if j is 0:
                            score_list.append(preds[ind, :sorted_sent_length_batch[ind]])
                        else:
                            score_list[ind] = torch.cat((score_list[ind], preds[ind, :sorted_sent_length_batch[ind]]),
                                                        dim=0)

                    scores, _ = pack_padded_sequence(scores[:len(sorted_sent_length_batch)], sorted_sent_length_batch,
                                                     batch_first=True)
                    targets, _ = pack_padded_sequence(targets[:len(sorted_sent_length_batch)], sorted_sent_length_batch,
                                                      batch_first=True)

                else:
                    sorted_sent_length_batch = sorted_sent_lengths[:, j]
                    sentence_batch_sorted = sorted_paragraphs[:, j]
                    targets = sentence_batch_sorted[:, 1:]
                    # For CIDEr scoring
                    for ind in range(targets.size(0)):
                        if j is 0:
                            paragraph_reference.append(targets[ind, :sorted_sent_length_batch[ind]])
                        else:
                            paragraph_reference[ind] = torch.cat((paragraph_reference[ind], targets[ind, :sorted_sent_length_batch[ind]]), dim=0)
                        #references.add(targets[ind, :sorted_sent_length_batch[ind]])

                    predictions_batch_sorted = predictions[:, j]
                    scores = predictions_batch_sorted

                    # For CIDEr scoring
                    _, preds = torch.max(scores, dim=2)
                    for ind in range(preds.size(0)):
                        #temp_preds.append(preds[ind][:sorted_sent_length_batch[ind]])  # remove pads
                        if j is 0:
                            score_list.append(preds[ind, :sorted_sent_length_batch[ind]])
                        else:
                            score_list[ind] = torch.cat((score_list[ind], preds[ind, :sorted_sent_length_batch[ind]]), dim=0)

                    scores, _ = pack_padded_sequence(scores[:len(sorted_sent_length_batch)], sorted_sent_length_batch,
                                                     batch_first=True)
                    targets, _ = pack_padded_sequence(targets[:len(sorted_sent_length_batch)], sorted_sent_length_batch,
                                                      batch_first=True)

                # a = sum(sorted_sent_length_batch)

                # Calculate loss
                loss_word_lstm = criterion(scores, targets)
                total_losses.append(loss_word_lstm)

            hypothesis.extend(score_list)
            references.extend(paragraph_reference)

            total_losses.append(tag_loss)
            total_losses.append(sentence_lstm_loss)

            total_losses = sum(total_losses)

            # Add doubly stochastic attention regularization
            total_losses += (1. - alphas_visual[:, :num_sents_minibatch[0]].sum(dim=1) ** 2).mean()
            total_losses += (1. - alphas_semantic[:, :num_sents_minibatch[0]].sum(dim=1) ** 2).mean()

            fifty_batch_loss += total_losses.item()
        if i % 10 is 0:
            print('Validation batch loss for batch {0}: {1}'.format(i, fifty_batch_loss))
    for i in range(len(hypothesis)):
        s = [vocab.idx2word[idx] for idx in hypothesis[i].tolist()]
        r = [vocab.idx2word[idx] for idx in references[i].tolist()]
        gts[i] = [' '.join(s)]
        res[i] = [' '.join(r)]

    #calculate cider score
    score, scores = cider_score_function.compute_score(gts, res)
    print('Mean CIDEr score for validation set {0}'.format(score))
    return score



if __name__ == '__main__':
    main()