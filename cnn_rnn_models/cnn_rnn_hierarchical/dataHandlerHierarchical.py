import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
import random
import json
import h5py

class dataset(data.Dataset):
    """Radiology Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, json_imgs, vocab, transform = None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.img_data = json_imgs
        self.vocab = vocab
        self.transform = transform


    def __getitem__(self, index):
        """Returns one data pair (images and paragraph)."""
        vocab = self.vocab
        paths = self.img_data[index]['file_paths'] #radiology
        num_sents = len(self.img_data[index]['paragraph'])
        paragraph = self.img_data[index]['paragraph']

        #IMAGE
        image_tensor = torch.zeros(len(paths), 3, 224, 224)

        for i, path in enumerate(paths):
            image = Image.open(path).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            image_tensor[i] = image

        #TAGS
        tags = []
        tags_yn = []
        if int(self.img_data[index]['Fracture']) > 0:
            tags.append('Fracture')
            tags_yn.append(1)
        else:
            tags_yn.append(0)
        if int(self.img_data[index]['Implant']) > 0:
            tags.append('Implant')
            tags_yn.append(1)
        else:
            tags_yn.append(0)
        if int(self.img_data[index]['Tumor']) > 0:
            tags.append('Tumor')
            tags_yn.append(1)
        else:
            tags_yn.append(0)
        if int(self.img_data[index]['Osteoarthritis']) > 0:
            tags.append('Osteoarthritis')
            tags_yn.append(1)
        else:
            tags_yn.append(0)
        if len(tags) is 0:
            tags.append('Normal')
            tags_yn.append(1)
        else:
            tags_yn.append(0)

        #tags = [self.img_data[index]['Fracture'], self.img_data[index]['Implant'], self.img_data[index]['Tumor'], self.img_data[index]['Osteoarthritis']]

        #Potential tags
        #'Filename', 'Exam_id', 'Rprt', 'split_sets', 'split_name', 'Fracture_oblique', 'Fracture_angulation', 'Exam_body_part', 'Exam_view', 'Side', 'Prev_frx', 'Pseudo_arthrosis', 'Dislocation', 'Implant_hip', 'Implant_knee',
        #'Implant_total_knee', 'Implant_hemi_knee', 'Implant_total_hip', 'Implant_hemi_hip', 'Implant_nail', 'Implant_plate', 'Implant_cerklage_wires', 'Implant_problem', 'Implant_loosening', 'Implant_external_fix', 'Tumor_nof', 'imgs',
        #'osteo_fg', 'Exam_type', 'Osteoarthritis_revisit', 'Osteoarthritis_other', 'Tumor_benign', 'Tumor_revisit', 'Tumor_infection', 'Other', 'Prosthetic', 'Fracture', 'Fracture_displaced', 'Fracture_undisplaced', 'Fracture_intraarticular',
        #'Fracture_spiral', 'Fracture_comminute', 'Osteoarthritis', 'Osteoarthritis_severe', 'Osteoarthritis_light', 'Implant', 'Tumor', 'Tumor_osteolytic', 'Tumor_sclerotic', 'Implant_fg', 'file_paths', 'paragraph'
        """
        fracture_tagnames_ankle = ['Fracture', 'Fracture_displaced', 'Fracture_undisplaced', 'Fracture_intraarticular', 'Fracture_spiral', 'Fracture_comminute', ]
        implant_tagnames_ankle = ['Implant', 'Implant_nail', 'Implant_plate', 'Implant_cerklage_wires', 'Implant_problem', 'Implant_loosening', 'Implant_external_fix', 'Implant_hip', 'Implant_knee',
        'Implant_total_knee', 'Implant_hemi_knee', 'Implant_total_hip', 'Implant_hemi_hip', 'Implant_fg']
        tumor_tagnames = ['Tumor', 'Tumor_osteolytic', 'Tumor_sclerotic', 'Tumor_benign', 'Tumor_revisit', 'Tumor_infection']
        """

        #PARAGRAPH
        allsents = []
        lengths = []
        for sent in paragraph:
            sent = str(sent).lower()
            sent = sent.replace("\n", "")
            tokens = nltk.tokenize.word_tokenize(str(sent).lower())
            caption = []
            caption.append(vocab('<start>'))
            caption.extend([vocab(token) for token in tokens])
            caption.append(vocab('<end>'))
            allsents.append(caption)
            lengths.append(len(caption))


        paragraph = torch.zeros(len(allsents), max(lengths)).long()
        #lengths_tensor = torch.tensor(lengths)
        for i, cap in enumerate(allsents):
            cap_torch = torch.Tensor(cap)
            end = lengths[i]
            paragraph[i, :end] = cap_torch[:end]

        return image_tensor, paragraph, lengths, tags, tags_yn


    def __len__(self):
        return len(self.img_data)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (images, paragraph, sentence_lengths).
            - image: torch tensor of shape (3, 256, 256).
            - paragraph: torch tensor of shape (?); variable length.
            - sentence_lengths: list of sentence_lengths
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    images, paragraph, sentence_lengths, tags, tags_yn = zip(*data)


    num_imgs = [len(img_tuple) for img_tuple in images]
    images_tensor = torch.zeros(len(images), max(num_imgs), 3, 224, 224)

    for i, images_tup in enumerate(images):
        images_tensor[i, :num_imgs[i], :, :, :] = images_tup

    max_length = [max(element) for element in sentence_lengths]
    num_sents = [len(element) for element in sentence_lengths]  #behövs en torch-tensor eller funkar list
    #num_sents = torch.Tensor(num_sents) # (batch_size)

    sentence_lengths_tensor = torch.zeros(len(sentence_lengths), max(num_sents)).long() #(batch_size, max_num_sents)
    for i, ele in enumerate(sentence_lengths):
        end = len(ele)
        sentence_lengths_tensor[i, :end] = torch.tensor(ele) # skippa tensor för längder?

    paragraph_tensor = torch.zeros(len(paragraph), max(num_sents), max(max_length)).long() #batch_size, max_num_sents, max_sent_length


    for i, tup_paragraph in enumerate(paragraph):
        for j, sent_length in enumerate(sentence_lengths[i]):
            paragraph_tensor[i, j, :sent_length] = tup_paragraph[j, :sent_length]

    #Work in progress
    #return list of paragraphs with same num of sents

    paragraph_list_tensor = []
    batch_sizes = []

    num_sents_torch = torch.Tensor(num_sents).long()
    num_sents_torch, sort_ind = num_sents_torch.sort(descending=True)
    num_sents.sort(reverse=True)
    num_sents_unique = list(set(num_sents))
    num_sents_unique.sort(reverse=True)

    paragraph = tuple(paragraph[i.item()] for i in sort_ind)
    sentence_lengths = tuple(sentence_lengths[i.item()] for i in sort_ind)
    images_tensor = images_tensor[sort_ind]
    sentence_lengths_tensor = sentence_lengths_tensor[sort_ind]

    cummulative_count = 0
    for i, ele in enumerate(num_sents_unique):
        count = num_sents.count(ele)
        batch_sizes.append(count)
        paragraph_tensor_fixed_num_sent = torch.zeros(count, ele, max(max_length)).long()  # batch_size, max_num_sents, max_sent_length
        for i, tup_paragraph in enumerate(paragraph[cummulative_count:(cummulative_count+count)]):
            for j, sent_length in enumerate(sentence_lengths[i+cummulative_count]):
                paragraph_tensor_fixed_num_sent[i, j, :sent_length] = tup_paragraph[j, :sent_length]
        paragraph_list_tensor.append(paragraph_tensor_fixed_num_sent)

        cummulative_count += count

    tags_yn_tensor = torch.zeros(len(tags_yn), len(tags_yn[0])).long()
    for i,tags in enumerate(tags_yn):
        for j, tag in enumerate(tags):
            tags_yn_tensor[i, j] = tag


    return images_tensor, paragraph_list_tensor, batch_sizes, tags, tags_yn_tensor, sentence_lengths_tensor, num_sents, max_length



def get_loader(json_imgs, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom Radiology dataset."""
    # Radiology report dataset
    dataSet = dataset(json_imgs = json_imgs, vocab = vocab, transform = transform)
    # Data loader for Radiology dataset
    # This will return (images, paragraphs, sentence_lengths, num_sents) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # paragraph: a tensor of shape (batch_size, sentence_length, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=dataSet,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader