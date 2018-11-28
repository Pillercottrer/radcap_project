from data_preprocessing.build_vocab import Vocabulary
import pickle
import numpy as np
import torch
import numpy as np
import torchvision.transforms as transforms
import pickle
import nltk
import json
from PIL import Image
from skimage.transform import resize
from imageio import imread
from cnn_rnn_models.cnn_rnn_hierarchical.model_hierarchical import Encoder, CoAttention, MLC, SentenceLSTMDecoder, Embedding, WordLSTMDecoder



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('hello world')


lengths = torch.Tensor([5,3,4,7,699,1]).long()
lenths_sort, sort_ind = torch.sort(lengths, descending=True)


reverse_sort = []
for i in range(len(lengths)):
    for j, ele in enumerate(sort_ind):
        if i is ele.item():
            reverse_sort.append(j)
            continue

reversed_questionsmark = lenths_sort[reverse_sort]

a = torch.Tensor([0,1,2,3]).long()

for i in range(max(a)):
    print(i)


radimgs = json.load(open('./data_preprocessing/radcap_bodypartsplit_data.json', 'r'))

for img in radimgs['ankle'][:10]:
    print(img['Fracture'])
    print(int(img['Fracture']) + 3)
    for sent in img['paragraph']:
        print(sent)





flickrimgs = json.load(open('./data_preprocessing/all_imgcap.json', 'r'))

image_path = radimgs['ankle'][0]['file_paths'][0]
paragraphs = flickrimgs[0]['captions']
# Load vocabulary wrapper
with open('./data/flickrvocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

paragraph_tokens =[]

for sent in paragraphs:
    tokens = nltk.tokenize.word_tokenize(str(sent).lower())
    print(tokens)
    caption = []
    caption.append(vocab('<start>'))
    caption.extend([vocab(token) for token in tokens])
    caption.append(vocab('<end>'))
    caption = torch.Tensor(caption)
    paragraph_tokens.append(caption)




lengths = [len(sent) for sent in paragraph_tokens]

target_paragraphs = torch.zeros(1, len(paragraph_tokens), max(lengths)).long().to(device)

for i, sent_tokens in enumerate(paragraph_tokens):
    target_paragraphs[0, i,:lengths[i]] = sent_tokens

print(len(target_paragraphs))

paragraph_sent_lengths = torch.Tensor(lengths).long()
paragraph_sent_lengths = paragraph_sent_lengths.unsqueeze(0).to(device)

num_sents = torch.Tensor([3]).to(device)


transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])



# Read image and process
img = imread(image_path)

if len(img.shape) == 2:
    img = img[:, :, np.newaxis]
    img = np.concatenate([img, img, img], axis=2)
img = resize(img, (256, 256), mode='constant')
img = img.transpose(2, 0, 1)
img = img / 255.
img = torch.FloatTensor(img).to(device)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
transform = transforms.Compose([normalize])
image = transform(img)  # (3, 256, 256)



#Constants
mlc_dim = 25 # num of tags
vocab_size = 11111 # random value for testing
semantic_att_embed_size = 100




#Encoding
encoder = Encoder()
encoder = encoder.to(device)

image = image.unsqueeze(0)  # (1, 3, 256, 256)
encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)

#set batch-size and encoder_dim
batch_size = encoder_out.size(0)
encoder_dim = encoder_out.size(-1)

# Flatten image
encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
num_pixels = encoder_out.size(1) #number of visual feature vectors
#Semantic features

embedding = Embedding(vocab_size)
embedding = embedding.to(device)

mlc = MLC(num_pixels, encoder_dim, embedding, vocab)
mlc = mlc.to(device)
semantic_features = mlc(encoder_out)


#Testing sentence topic generator (håll tummarna)
sent_lstm = SentenceLSTMDecoder(vocab_size)
sent_lstm.to(device)
topic_tensor, stop_tensor = sent_lstm(encoder_out, semantic_features)

#topic_tensor = torch.Tensor(len(topic_vectors), 111)
#topic_tensor = torch.cat(topic_vectors)
#topic_tensor = topic_tensor.unsqueeze(0)

word_lstm = WordLSTMDecoder(vocab_size, embedding)
word_lstm.to(device)
predictions = word_lstm(topic_tensor, num_sents, target_paragraphs, paragraph_sent_lengths)
print('break')






"""
# Flatten encoding
encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
num_pixels = encoder_out.size(1)

#Multi label classification network
mlc = MLC(num_pixels,encoder_dim, mlc_dim, vocab_size, semantic_att_embed_size) #visual_attention dim, encoder_dim, tag_dim. vocab_size, embed_dim
mlc = mlc.to(device)

mlc.init_weights()
tags = mlc(encoder_out)

#Embed words
embed = Embedding(vocab_size, semantic_att_embed_size)
embed = embed.to(device)
a_att = embed(tags)


#Co-Attention
h = torch.zeros(512).to(device)
h = h.unsqueeze(0)

#visual_features = encoder_out.contiguous().view(-1, encoder_dim * num_pixels)

coAttention = CoAttention(encoder_dim, semantic_att_embed_size, 100, 512, 100)
coAttention.to(device)
ctx = coAttention(encoder_out, a_att, h)
"""