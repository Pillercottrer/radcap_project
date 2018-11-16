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

class FlickrDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json_imgs, vocab, transform = None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform
        """

        """
        f = h5py.File('data.h5', 'r')
        self.label_start_ix = f['label_start_ix']
        self.label_end_ix = f['label_end_ix']
        self.labels = f['labels']
        self.images = f['images']
        self.label_length = f['label_length']
        """
        self.img_data = json_imgs
        self.vocab = vocab
        self.transform = transform
        self.root = root


    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        #print(index)
        #start_capindex = self.label_start_ix[index]
        #end_capindex = self.label_end_ix[index]
        vocab = self.vocab
        #print(start_capindex.dtype)
        #print(end_capindex.dtype)
        #print('startcap: %d' % start_capindex)
        #print('endcap: %d' % end_capindex)
        #print('endcap - startcap: %d' % (end_capindex - start_capindex))
        #num_of_caps = end_capindex - start_capindex
        #cap_ind = np.randint(start_capindex, end_capindex)
        #print('cap_ind %d' % cap_ind)
        #caption = self.labels[cap_ind,:]
        #image = self.images[index]

        #path = self.img_data[index]['id'] + '.jpg'

        num_imgs = len(self.img_data[index]['file_paths'])
        path = self.img_data[index]['file_paths'][random.randint(0,num_imgs-1)]
        num_cap = len(self.img_data[index]['captions'])
        caption = self.img_data[index]['captions'][random.randint(0,num_cap-1)]

        #image = Image.open(os.path.join(self.root, path)).convert('RGB')
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)

        #caption_torch = torch.from_numpy(caption.astype(float))
        #image_torch = torch.from_numpy(image.astype(float))

        """
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target
        """
        return image, target

    def __len__(self):
        return len(self.img_data)



def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

def make_weights_for_balanced_classes(json_imgs, nclasses):
    count = [0] * nclasses
    for item in json_imgs:
        if item['fracture'] == str(1): #hard coded classes '0' and '1'
            count[1] += 1
        else:
            count[0] += 1

    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(json_imgs)

    for idx, img in enumerate(json_imgs):
        if img['fracture'] == 1:
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
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths


def get_loader(root, json_imgs, vocab, transform, batch_size, shuffle, num_workers, ):
    """Returns torch.utils.data.DataLoader for custom Flicker8k dataset."""
    # Flicker8k caption dataset
    flickr = FlickrDataset(root = root, json_imgs = json_imgs, vocab = vocab, transform = transform)
    # Data loader for Flicker dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).

    weights = make_weights_for_balanced_classes(json_imgs, 2)
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    data_loader = torch.utils.data.DataLoader(dataset=flickr,
                                              batch_size=batch_size,
                                              sampler=sampler,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader