from collections import Counter, defaultdict
from gensim.models import Word2Vec
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from torch import nn
from torch.autograd import Variable
from torchvision import models, transforms

import json
import numpy as np
import random
import torch
import torch.nn.functional as F
import ABot
import json
import h5py

#Load Data
chat_path = '../chat_processed_data.h5'
chat_params = '../chat_processed_params.json'
f = h5py.File(chat_path, 'r')
with open(chat_params, 'rb') as input_json:
	info = json.loads(input_json.read().decode('utf-8'))
unique_img_train =  info['unique_img_train']
unique_img_val =   info['unique_img_val']
ind2word = info['ind2word']
ind2word = {int(key): value for key,value in ind2word.items()}
word2ind = info['word2ind']
unique_img_test =   info['unique_img_test']

#Add tokens
START_TOKEN = '<START>'
END_TOKEN = '<END>' 
PAD_TOKEN = '<PAD>'
START_INDEX = len(ind2word.keys()) + 1
END_INDEX = len(ind2word.keys()) + 2
PAD_INDEX = 0

ind2word[PAD_INDEX] = PAD_TOKEN
ind2word[START_INDEX] = START_TOKEN
ind2word[END_INDEX] = END_TOKEN

word2ind[PAD_TOKEN] = PAD_INDEX
word2ind[START_TOKEN] = START_INDEX
word2ind[END_TOKEN] = END_INDEX

#CUDA
USE_CUDA = False
gpu = 0

#Parameters
batch_first = False
num_layers = 1
hidden_dim = 512
embed_size = 300
embedding_weights = np.random.random((len(ind2word.keys()),embed_size))
vocab_size = embedding_weights.shape[0]
embedding_size = embedding_weights.shape[1]
vgg_out = 4096
batch_size = 200

#Define Models
vgg = ABot.VGG()
embedding_layer = ABot.EmbeddingLayer(embedding_weights)
question_encoder = ABot.QuestionEncoder(embedding_size, hidden_dim, num_layers=num_layers, batch_first=batch_first)
fact_encoder = ABot.FactEncoder(embedding_size*2, hidden_dim, num_layers=num_layers, batch_first=batch_first)
history_encoder = ABot.HistoryEncoder(hidden_dim*2 + vgg_out, hidden_dim, num_layers=num_layers, batch_first=batch_first)
answer_decoder = ABot.AnswerDecoder(embedding_size, hidden_dim, vocab_size, num_layers=num_layers, batch_first=batch_first)

if USE_CUDA:
	vgg = vgg.cuda(gpu)
	embedding_layer = embedding_layer.cuda(gpu)
	question_encoder = question_encoder.cuda(gpu)
	fact_encoder = fact_encoder.cuda(gpu)
	history_encoder = history_encoder.cuda(gpu)
	answer_decoder = answer_decoder.cuda(gpu)


def supervisedTrain(question_batch, question_batch_len, answer_batch, answer_batch_len, vgg, embedding_layer, question_encoder, fact_encoder, history_encoder, answer_decoder, num_dialogs=10):
	question_batch = question_batch.astype(np.int)
	answer_batch = answer_batch.astype(np.int)
	question_batch_len = question_batch_len.astype(np.int)
	answer_batch_len = answer_batch_len.astype(np.int)
	for turn in range(num_dialogs):
		current_question = question_batch[:,turn]
		current_question_length = question_batch_len[:,turn]
		current_question_length_tensor = torch.from_numpy(current_question_length).long()
		current_question_tensor = Variable(torch.from_numpy(current_question).long())
		if USE_CUDA:
			current_question_tensor = current_question_tensor.cuda(gpu)
			current_question_length = current_question_length.cuda(gpu)

		#Embedding Layer
		current_question_embedding = embedding_layer(current_question_tensor)

		#Question Encoder
		current_question_length_tensor_sorted, perm_idx = current_question_length_tensor.sort(0, descending=True)
		current_question_embedding = current_question_embedding[perm_idx]
		current_question_embedding = current_question_embedding.transpose(0,1)
		question_encoder_hidden, question_encoder_memory = question_encoder.initHidden(batch_size=len(current_question))
		if USE_CUDA:
			question_encoder_hidden = question_encoder_hidden.cuda(gpu)
			question_encoder_memory = question_encoder_memory.cuda(gpu)
		print (current_question_length_tensor_sorted)
		print (current_question_embedding)
		packed_current_question_embedding = nn.utils.rnn.pack_padded_sequence(current_question_embedding, current_question_length_tensor_sorted.cpu().numpy())
		packed_current_question_encoding, question_encoder_hidden, question_encoder_memory = question_encoder(packed_current_question_embedding, question_encoder_hidden, question_encoder_memory)
		current_question_encoding, _ = nn.utils.rnn.pad_packed_sequence(packed_current_question_encoding)
		print (current_question_encoding)
		sys.exit()

# for i in f['ans_train']:
# 	if len(i) == 10:
# 		print (i==0)
# 		break
supervisedTrain(f['ques_train'][:20], f['ques_length_train'][:20], f['ans_train'][:20], f['ans_length_train'][:20], vgg, embedding_layer, question_encoder, fact_encoder, history_encoder, answer_decoder)