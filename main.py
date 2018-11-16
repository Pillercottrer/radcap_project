import json
import random
import torch
import numpy as np
import nltk
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from PIL import Image
from sample import main

with open('/home/emil/Python_projects/Jupyter notebooks/data_file_captions.json') as json_file:
    data = json.load(json_file)

print(len(data))

with open('data.json') as json_file:
    data = json.load(json_file)

index = 4
vocab = data['ix_to_word']
print(vocab)

print(torch.cuda.is_available())


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

with open('test_imgcap.json') as json_file:
    json_data = json.load(json_file)


for i in range(10)[:3]:
    print(i)

val = json.load(open('./data/annotations/captions_val2017.json', 'r'))
train = json.load(open('./data/annotations/captions_train2017.json', 'r'))

"""

cocoimgs = json.load(open('./coco_raw.json', 'r'))


print(cocoimgs[0].keys())
print(len(cocoimgs))
print(cocoimgs[0]['file_path'])
print(cocoimgs[0]['id'])
print(cocoimgs[0]['captions'])


print(len(cocoimgs))
print(int(round(len(cocoimgs)*0.8)))
print(len(cocoimgs)*0.2)


"""



