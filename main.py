import json
import h5py
import random
import torch
import numpy as np
import nltk
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from PIL import Image

with open('/home/emil/Python_projects/Jupyter notebooks/data_file_captions.json') as json_file:
    data = json.load(json_file)

print(len(data))

f = h5py.File('data.h5', 'r')
print(list(f.keys()))

print(f['images'][0].shape)
print(f['images'][0])
print(f['labels'].shape)
print(len(f['labels']))
print(f['labels'][0,:])

with open('data.json') as json_file:
    data = json.load(json_file)


vocab = data['ix_to_word']
print(vocab)

for i in f['labels'][0,:]:
    print(vocab.get(str(i)))

index = 4

print(f['label_start_ix'][index])
print(f['label_end_ix'][index])
cap_num = f['label_end_ix'][index] - f['label_start_ix'][index]

cap_ind = random.randint(0,cap_num)
caption = f['labels'][f['label_start_ix'][index],:]
caption2 = f['labels'][f['label_start_ix'][index]+cap_ind,:]

for i in caption:
    print(vocab.get(str(i)))

for i in caption2:
    print(vocab.get(str(i)))

print(torch.cuda.is_available())



caption = f['labels'][cap_ind,:]


print(caption.dtype)
target = torch.from_numpy(caption.astype(float))
print(target)
image = torch.from_numpy(f['images'][index].astype(float))
print(image)

print(torch.__version__)
print(torch.rand(2, 3))

with open('all_imgcap.json') as json_file:
    json_data = json.load(json_file)
caption = json_data[index]['captions'][0]
tokens = nltk.tokenize.word_tokenize(str(caption).lower())
print(caption)
print(tokens)


transform = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

image = Image.open(json_data[index]['filepath']).convert('RGB')
image = transform(image)

print(image)

print(json_data[index]['filepath'])
print(json_data[index]['id'] + '.jpg')

