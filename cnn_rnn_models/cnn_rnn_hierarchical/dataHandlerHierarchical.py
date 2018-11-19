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
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json_imgs, vocab, transform = None):
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
        self.root = root


    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        vocab = self.vocab
        paths = self.img_data[index]['file_paths'] #radiology
        num_sents = len(self.img_data[index]['captions'])
        paragraph = self.img_data[index]['captions']

        #image = Image.open(os.path.join(self.root, path)).convert('RGB')
        image = Image.open(paths[0]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        image_tensor = torch.zeros(len(paths), 3, 224, 224)

        for i, path in enumerate(paths):
            image = Image.open(path).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            image_tensor[i] = image

        rad_combined_tensor, _ = torch.max(image_tensor, 0)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        lengths = []
        if self.img_data[index]['split'] == 'val':
            allcaps = []
            for cap in self.img_data[index]['captions']:
                tokens = nltk.tokenize.word_tokenize(str(cap).lower())
                caption = []
                caption.append(vocab('<start>'))
                caption.extend([vocab(token) for token in tokens])
                caption.append(vocab('<end>'))
                allcaps.append(caption)
                lengths.append(len(caption))
            all_captions = torch.zeros(len(allcaps), max(lengths)).long()
            for i, cap in enumerate(allcaps):
                cap_torch = torch.Tensor(cap)
                end = lengths[i]
                all_captions[i, :end] = cap_torch[:end]
            return image, target, all_captions
        else:
            return image, target

    def __len__(self):
        return len(self.img_data)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    counter = 0
    for it in zip(*data):
        if counter == 0:
            images = it
        elif counter == 1:
            captions = it
        elif counter == 2:
            all_captions = it
        elif counter == 3:
            num_caps = it
        counter += 1

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]


    if(counter < 3):
        return images, targets, lengths
    else:
        # Merge all_caps lol
        allcap_lengths = []
        num_caps = 0
        for tup in all_captions:
            if len(tup) > num_caps:
                num_caps = len(tup)
            for ele in tup:
                allcap_lengths.append(len(ele))
                break

        all_cap_tensor = torch.zeros(len(all_captions), max(allcap_lengths), num_caps).long()

        for i, caps in enumerate(all_captions):
            end = lengths[i]
            for j, cap in enumerate(caps):
                all_cap_tensor[i,:end,j] = cap[:end]

        return images, targets, lengths, all_cap_tensor

def merge_1d_to_2d(tensor):
    # Merge (from tuple of 1D tensor to 2D tensor).
    lengths = [len(ele) for ele in tensor]
    targets = torch.zeros(len(tensor), max(lengths)).long()
    for i, cap in enumerate(tensor):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return targets



def get_loader(root, json_imgs, vocab, transform, batch_size, shuffle, num_workers, ):
    """Returns torch.utils.data.DataLoader for custom Flicker8k dataset."""
    # Flicker8k caption dataset
    dataSet = dataset(root = root, json_imgs = json_imgs, vocab = vocab, transform = transform)
    # Data loader for Flicker dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=dataSet,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader