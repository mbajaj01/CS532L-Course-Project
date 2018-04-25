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
import time
import dataloader
import math
import ABot_Encoder
import ABot_Decoder
import QBot
import QBot_Encoder
import QBot_Decoder


class ImageEncoder(nn.Module):
	def __init__(self, input_dim, output_dim, dropout=0.5):
		super(ImageEncoder, self).__init__()
		self.dropout = nn.Dropout()
		self.linear = nn.Linear(input_dim, output_dim)

	def forward(self, image):
		return self.linear(self.dropout(image))

class EmbeddingLayer(nn.Module):
	def __init__(self, embedding_weights):
		super(EmbeddingLayer, self).__init__()
		self.embedding_layer = nn.Embedding(num_embeddings=embedding_weights.shape[0], embedding_dim=embedding_weights.shape[1], padding_idx=0)
		self.embedding_layer.weight.data.copy_(torch.from_numpy(embedding_weights))

	def forward(self, X):
		return self.embedding_layer(X)

class Encoder(nn.Module):
	def __init__(self, input_dim, output_dim, rnn='LSTM', num_layers=1, batch_first=True):
		super(Encoder, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.rnn = rnn
		self.num_layers = num_layers
		self.lstm = getattr(nn, rnn)(self.input_dim, self.output_dim, num_layers)

	def forward(self, X, hidden_state):
		lstm_out, hidden_state = self.lstm(X, hidden_state)
		return lstm_out, hidden_state

	def initHidden(self, batch_size):
		weight = next(self.parameters()).data
		if self.rnn == 'LSTM':
			return (Variable(weight.new(self.num_layers, batch_size, self.output_dim).zero_()), Variable(weight.new(self.num_layers, batch_size, self.output_dim).zero_()))
		else:
			return Variable(weight.new(self.num_layers, batch_size, self.output_dim).zero_())


class Discriminator(nn.Module):
    def __init__(self, params, embedding_weights):
        super(Discriminator, self).__init__()
        self.params = params
        self.question_encoder = Encoder(params['embedding_size'], params['hidden_dim'], num_layers=params['num_layers'], rnn=params['rnn_type'], batch_first=params['batch_first'])
        self.answer_encoder = Encoder(params['embedding_size'], params['hidden_dim'], num_layers=params['num_layers'], rnn=params['rnn_type'], batch_first=params['batch_first'])
        self.image_encoder = ImageEncoder(params['vgg_out'], params['image_embed_size'])
        self.embedding_layer = EmbeddingLayer(embedding_weights)
        self.linear1 = nn.Linear(2*params['hidden_dim'] + params['image_embed_size'], params['hidden_dim'])
        self.linear2 = nn.Linear(params['hidden_dim'], 1)

    def forward(self, question_batch, question_batch_len, answer_batch, answer_batch_len, image_batch, batch_mode=True, mode='index'):
        if mode == 'index':
            question_batch_embedding = self.embedding_layer(question_batch)
        else:
            question_batch_embedding = torch.mm(question_batch.view(-1, self.params['vocab_size']),self.embedding_layer.embedding_layer.weight).view(question_batch.size(0), question_batch.size(1), -1)
        batchSize = question_batch_embedding.size(0)

		#Question Encoder
        sort_index = 0 if self.params['batch_first'] else 1
        question_batch_encoder_hidden = self.question_encoder.initHidden(batch_size=question_batch_embedding.size(0))
        if batch_mode:
            question_batch_length_tensor_sorted, question_perm_idx = question_batch_len.sort(0, descending=True)
            _, question_perm_idx_resort = question_perm_idx.sort(0, descending=False)
            question_batch_embedding = question_batch_embedding.index_select(0,question_perm_idx)
            if not self.params['batch_first']:
                question_batch_embedding = question_batch_embedding.transpose(0,1)
            packed_question_batch_embedding = nn.utils.rnn.pack_padded_sequence(question_batch_embedding, question_batch_length_tensor_sorted.data.cpu().numpy(), batch_first=self.params['batch_first'])
            packed_question_batch_encoding, question_batch_encoder_hidden = self.question_encoder(packed_question_batch_embedding, question_batch_encoder_hidden)
            output_question_batch_encoding, _ = nn.utils.rnn.pad_packed_sequence(packed_question_batch_encoding)
            output_question_batch_encoding = output_question_batch_encoding.index_select(sort_index,question_perm_idx_resort)
            if self.params['rnn_type'] == 'LSTM':
                question_batch_encoder_hidden = (question_batch_encoder_hidden[0].index_select(1, question_perm_idx_resort), question_batch_encoder_hidden[1].index_select(1, question_perm_idx_resort))
            else:
                question_batch_encoder_hidden = question_batch_encoder_hidden.index_select(1, question_perm_idx_resort)
        else:
            if not self.params['batch_first']:
                question_batch_embedding = question_batch_embedding.transpose(0,1)
            output_question_batch_encoding, question_batch_encoder_hidden = self.question_encoder(question_batch_embedding, question_batch_encoder_hidden)

        question_batch_encoding = question_batch_encoder_hidden[0].narrow(0, self.params['num_layers']-1,1)
        question_batch_encoding = question_batch_encoding.transpose(0,1)

        #Answer Encoder
        if mode == 'index':
            answer_batch_embedding = self.embedding_layer(answer_batch)
        else:
            answer_batch_embedding = torch.mm(answer_batch.view(-1, self.params['vocab_size']),self.embedding_layer.embedding_layer.weight).view(answer_batch.size(0), answer_batch.size(1), -1)
        answer_batch_encoder_hidden = self.answer_encoder.initHidden(batch_size=answer_batch_embedding.size(0))
        if batch_mode:
            answer_batch_length_tensor_sorted, answer_perm_idx = answer_batch_len.sort(0, descending=True)
            _, answer_perm_idx_resort = answer_perm_idx.sort(0, descending=False)
            answer_batch_embedding = answer_batch_embedding.index_select(0,answer_perm_idx)
            if not self.params['batch_first']:
                answer_batch_embedding = answer_batch_embedding.transpose(0,1)
            packed_answer_batch_embedding = nn.utils.rnn.pack_padded_sequence(answer_batch_embedding, answer_batch_length_tensor_sorted.data.cpu().numpy(), batch_first=self.params['batch_first'])
            packed_answer_batch_encoding, answer_batch_encoder_hidden = self.answer_encoder(packed_answer_batch_embedding, answer_batch_encoder_hidden)
            output_answer_batch_encoding, _ = nn.utils.rnn.pad_packed_sequence(packed_answer_batch_encoding)
            output_answer_batch_encoding = output_answer_batch_encoding.index_select(sort_index,answer_perm_idx_resort)
            if self.params['rnn_type'] == 'LSTM':
                answer_batch_encoder_hidden = (answer_batch_encoder_hidden[0].index_select(1, answer_perm_idx_resort), answer_batch_encoder_hidden[1].index_select(1, answer_perm_idx_resort))
            else:
                answer_batch_encoder_hidden = answer_batch_encoder_hidden.index_select(1, answer_perm_idx_resort)
        else:
            if not self.params['batch_first']:
                answer_batch_embedding = answer_batch_embedding.transpose(0,1)
            output_answer_batch_encoding, answer_batch_encoder_hidden = self.answer_encoder(answer_batch_embedding, answer_batch_encoder_hidden)

        answer_batch_encoding = answer_batch_encoder_hidden[0].narrow(0, self.params['num_layers']-1,1)
        answer_batch_encoding = answer_batch_encoding.transpose(0,1)

        #Image Encoder
        image_batch_encoding = self.image_encoder(image_batch)

        combined_embedding = torch.cat((question_batch_encoding.squeeze(1), answer_batch_encoding.squeeze(1), image_batch_encoding), 1)
        output = F.sigmoid(self.linear2(F.tanh(self.linear1(F.tanh(combined_embedding)))))
        return output
