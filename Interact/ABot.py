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

class Attention(nn.Module):
	def __init__(self, question_dim, fact_dim, hidden_dim, filter_dim, image_dim, image_feature, output_dim):
		super(Attention, self).__init__()
		self.question_dim = question_dim
		self.fact_dim = fact_dim
		self.hidden_dim = hidden_dim
		self.filter_dim = filter_dim
		self.image_dim = image_dim
		self.output_dim = output_dim
		self.image_feature = image_feature
		self.question_linear = nn.Linear(self.question_dim, self.hidden_dim)
		self.fact_linear = nn.Linear(self.fact_dim, self.hidden_dim)
		self.filter_linear = nn.Linear(self.hidden_dim*2, self.filter_dim**2)
		self.output_linear = nn.Linear((self.image_dim - self.filter_dim + 1)**2*self.image_feature, self.output_dim)

	def forward(self, cnn_features, question_encoding, fact_encoding):
		question_hidden = self.question_linear(question_encoding)
		fact_hidden = self.fact_linear(fact_encoding)
		final_hidden = F.tanh(torch.cat((question_hidden, fact_hidden), 2))
		cnn_filter = self.filter_linear(final_hidden)
		cnn_filter = cnn_filter.view(-1, self.filter_dim, self.filter_dim)
		groups = cnn_features.size(0)
		cnn_features = cnn_features.permute(3,0,1,2)
		attended_features = F.conv2d(cnn_features, cnn_filter.unsqueeze(1), groups=groups)
		attended_features_reshaped = attended_features.permute(1,2,3,0).contiguous().view(groups, -1)
		output_features = self.output_linear(attended_features_reshaped).unsqueeze(0)
		return output_features

class DecoderAttention(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(DecoderAttention, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.linear1 = nn.Linear(input_dim, input_dim)
		self.linear2 = nn.Linear(input_dim, 1)
		self.linear3 = nn.Linear(input_dim, output_dim)

	def forward(self, X, indices, dropout=0.5):
		mask = indices == 0
		attention = self.linear2(F.dropout(F.tanh(self.linear1(X)), dropout, training=self.training)).squeeze(2)
		attention.masked_fill_(mask, -99999)
		weights = F.softmax(attention, dim=1)
		attended_weights = torch.mul(X, weights.unsqueeze(2).expand_as(X))
		output = self.linear3(torch.sum(attended_weights, 1))
		return output

class HistoryAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(HistoryAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, history, question, dropout=0.5):
        combined_embedding = F.tanh(history + question.expand_as(history))
        attention = F.softmax(self.linear(F.dropout(combined_embedding, dropout, training=self.training).view(-1, self.input_dim)).view(-1, history.size(1)), dim=1)
        attended_history = torch.bmm(attention.view(-1, 1, history.size(1)), history)
        return attended_history

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

class QuestionEncoder(nn.Module):
	def __init__(self, input_dim, output_dim, rnn='LSTM', num_layers=1, batch_first=True):
		super(QuestionEncoder, self).__init__()
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

class FactEncoder(nn.Module):
	def __init__(self, input_dim, output_dim, rnn='LSTM', num_layers=1, batch_first=True):
		super(FactEncoder, self).__init__()
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

class HistoryEncoder(nn.Module):
	def __init__(self, input_dim, output_dim, rnn='LSTM', num_layers=1, batch_first=True):
		super(HistoryEncoder, self).__init__()
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


class AnswerDecoder(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, rnn='LSTM', num_layers=1, batch_first=True):
		super(AnswerDecoder, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.rnn = rnn
		self.lstm = getattr(nn, rnn)(self.input_dim, self.hidden_dim, num_layers)
		self.linear = nn.Linear(self.hidden_dim, self.output_dim)

	def forward(self, X, hidden_state, isPacked=True):
		lstm_out, hidden_state = self.lstm(X, hidden_state)
		if isPacked:
			# lstm_out, temp = nn.utils.rnn.pad_packed_sequence(lstm_out)
			return lstm_out, hidden_state
		else:
			linear_out = self.linear(lstm_out)
			return linear_out, hidden_state

	def initHidden(self, batch_size):
		weight = next(self.parameters()).data
		if self.rnn == 'LSTM':
			return (Variable(weight.new(self.num_layers, batch_size, self.hidden_dim).zero_()), Variable(weight.new(self.num_layers, batch_size, self.hidden_dim).zero_()))
		else:
			return Variable(weight.new(self.num_layers, batch_size, self.hidden_dim).zero_())


class GumbelSampler(nn.Module):
	def __init__(self):
		super(GumbelSampler, self).__init__()

	def forward(self, input, noise, temperature=0.5):
		eps = 1e-20
		noise.data.add_(eps).log_().neg_()
		noise.data.add_(eps).log_().neg_()
		y = (input + noise) / temperature
		y = F.softmax(y, dim=1)
		max_val, max_idx = torch.max(y, y.dim()-1)
		y_hard = y == max_val.view(-1,1).expand_as(y)
		y = (y_hard.float() - y).detach() + y

		return y, max_idx.view(1, -1)#, log_prob
