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


class VGG(nn.Module):
	def __init__(self):
		super(VGG, self).__init__()
		self.vgg = models.vgg16(pretrained=True)
		self.vgg.classifier = nn.Sequential(*(self.vgg.classifier[i] for i in range(len(self.vgg.classifier) - 1)))

	def forward(self, image):
		return self.vgg(image)

class EmbeddingLayer(nn.Module):
	def __init__(self, embedding_weights):
		super(EmbeddingLayer, self).__init__()
		self.embedding_layer = nn.Embedding(num_embeddings=embedding_weights.shape[0], embedding_dim=embedding_weights.shape[1], padding_idx=0)
		self.embedding_layer.weight.data.copy_(torch.from_numpy(embedding_weights))

	def forward(self, X):
		return self.embedding_layer(X)

class QuestionEncoder(nn.Module):
	def __init__(self, input_dim, output_dim, num_layers=1, batch_first=True):
		super(QuestionEncoder, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.lstm = nn.LSTM(self.input_dim, self.output_dim, num_layers)

	def forward(self, X, hidden_state, memory):
		lstm_out, (hidden_state, memory) = self.lstm(X, (hidden_state, memory))
		return lstm_out, hidden_state, memory

	def initHidden(self, batch_size=1, max_seq_len=1):
		hidden_state = Variable(torch.zeros(max_seq_len, batch_size, self.output_dim))
		memory = Variable(torch.zeros(max_seq_len, batch_size, self.output_dim))
		return (hidden_state, memory)

class FactEncoder(nn.Module):
	def __init__(self, input_dim, output_dim, num_layers=1, batch_first=True):
		super(FactEncoder, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.lstm = nn.LSTM(self.input_dim, self.output_dim, num_layers)

	def forward(self, question, answer, hidden_state, memory):
		X = torch.cat((question, answer), dim=2)
		lstm_out, (hidden_state, memory) = self.lstm(X, (hidden_state, memory))
		return lstm_out, hidden_state, memory

	def initHidden(self, batch_size=1, max_seq_len=1):
		hidden_state = Variable(torch.zeros(max_seq_len, batch_size, self.output_dim))
		memory = Variable(torch.zeros(max_seq_len, batch_size, self.output_dim))
		return (hidden_state, memory)

class HistoryEncoder(nn.Module):
	def __init__(self, input_dim, output_dim, num_layers=1, batch_first=True):
		super(HistoryEncoder, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.lstm = nn.LSTM(self.input_dim, self.output_dim, num_layers)

	def forward(self, question_encoding, image_encoding, fact_encoding, hidden_state, memory):
		X = torch.cat((question_encoding, image_encoding, fact_encoding), dim=2)
		lstm_out, (hidden_state, memory) = self.lstm(X, (hidden_state, memory))
		return lstm_out, hidden_state, memory

	def initHidden(self, batch_size=1, max_seq_len=1):
		hidden_state = Variable(torch.zeros(max_seq_len, batch_size, self.output_dim))
		memory = Variable(torch.zeros(max_seq_len, batch_size, self.output_dim))
		return (hidden_state, memory)


class AnswerDecoder(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, batch_first=True):
		super(AnswerDecoder, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.hidden_dim = hidden_dim
		self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers)
		self.linear = nn.Linear(self.hidden_dim, self.output_dim)

	def forward(self, X, hidden_state, memory):
		lstm_out, (hidden_state, memory) = self.lstm(X, (hidden_state, memory))
		linear_out = self.linear(lstm_out)
		return linear_out, hidden_state, memory

	def initHidden(self, batch_size=1, max_seq_len=1):
		hidden_state = Variable(torch.zeros(max_seq_len, batch_size, self.hidden_dim))
		memory = Variable(torch.zeros(max_seq_len, batch_size, self.hidden_dim))
		return (hidden_state, memory)

