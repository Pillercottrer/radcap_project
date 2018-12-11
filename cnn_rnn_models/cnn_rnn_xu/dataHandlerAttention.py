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
        #path = self.img_data[index]['id'] + '.jpg'
        #path = self.img_data[index]['filepath'] #Flickr
        paths = self.img_data[index]['file_paths'] #radiology
        #num_cap = len(self.img_data[index]['captions'])
        #caption = self.img_data[index]['captions'][random.randint(0, num_cap-1)]

        #image = Image.open(os.path.join(self.root, path)).convert('RGB')
        image_tensor = torch.zeros(len(paths), 3, 224, 224)

        for i, path in enumerate(paths):
            image = Image.open(path).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            image_tensor[i] = image

        #rad_combined_tensor, _ = torch.max(image_tensor, 0)
        paragraph = []
        for i, sent in enumerate(self.img_data[index]['paragraph']):
            tokens = nltk.tokenize.word_tokenize(str(sent).lower())
            sentence = []
            if i is 0:
                sentence.append(vocab('<start>'))
            sentence.extend([vocab(token) for token in tokens])
            sentence.append(vocab('.'))
            paragraph.extend(sentence)
            #target = torch.Tensor(caption)
        test = vocab('<end>')
        paragraph.append(vocab('<end>'))
        target = torch.Tensor(paragraph)
        return image_tensor, target


        lengths = []
        if self.img_data[index]['split'] == 'val':
            allcaps = []
            for cap in self.img_data[index]['paragraph']:
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

def make_weights_for_balanced_classes(json_imgs, nclasses):
    count = [0] * nclasses
    for item in json_imgs:
        if int(item['Fracture']) > 0: #hard coded classes '0' and '1'
            count[1] += 1
        else:
            count[0] += 1

    weight_per_class = [0.] * nclasses
    """
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    """
    weight_per_class[0] = 0.001
    weight_per_class[1] = 100
    weight = [0] * len(json_imgs)

    for idx, img in enumerate(json_imgs):
        if int(img['Fracture']) > 0:
            weight[idx] = weight_per_class[1]
        else:
            weight[idx] = weight_per_class[0]
    return weight

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
    #images = torch.stack(images, 0)

    num_imgs = [len(img_tuple) for img_tuple in images]
    images_tensor = torch.zeros(len(images), max(num_imgs), 3, 224, 224)

    for i, images_tup in enumerate(images):
        images_tensor[i, :num_imgs[i], :, :, :] = images_tup

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]


    if(counter < 3):
        return images_tensor, targets, lengths
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



def get_loader(root, json_imgs, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom Flicker8k dataset."""
    # Radiological caption dataset
    dataSet = dataset(root = root, json_imgs = json_imgs, vocab = vocab, transform = transform)
    # Data loader for Radiology dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).

    #weights = make_weights_for_balanced_classes(json_imgs, 2)
    #weights = torch.DoubleTensor(weights)

    #sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    data_loader = torch.utils.data.DataLoader(dataset=dataSet,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader