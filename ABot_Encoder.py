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

class ABotEncoder(nn.Module):
	def __init__(self, params):
		super(ABotEncoder, self).__init__()
		self.params = params
		self.image_encoder = ABot.ImageEncoder(params['vgg_out'], params['image_embed_size'])
		self.question_encoder = ABot.QuestionEncoder(params['embedding_size'], params['hidden_dim'], num_layers=params['num_layers'], rnn=params['rnn_type'], batch_first=params['batch_first'])
		self.fact_encoder = ABot.FactEncoder(params['embedding_size'], params['hidden_dim'], num_layers=params['num_layers'], rnn=params['rnn_type'], batch_first=params['batch_first'])
		self.history_encoder = ABot.HistoryEncoder(params['hidden_dim']*2 + params['image_embed_size'], params['hidden_dim'], num_layers=params['num_layers'], rnn=params['rnn_type'], batch_first=params['batch_first'])

	def forward(self, question_batch_embedding, question_batch_len, fact_batch_embedding, fact_batch_len, image_batch, batch_mode=True):
	# def forward(self, question_batch_embedding, question_batch_len, fact_batch_embedding, fact_batch_len, image_batch, batch_mode=True):
		#Question Encoder
		question_batch_encoder_hidden = self.question_encoder.initHidden(batch_size=question_batch_embedding.size(0))

		if batch_mode:
			question_batch_length_tensor_sorted, question_perm_idx = question_batch_len.sort(0, descending=True)
			_, question_perm_idx_resort = question_perm_idx.sort(0, descending=False)
			question_batch_embedding = question_batch_embedding[question_perm_idx]
			if not self.params['batch_first']:
				question_batch_embedding = question_batch_embedding.transpose(0,1)
			packed_question_batch_embedding = nn.utils.rnn.pack_padded_sequence(question_batch_embedding, question_batch_length_tensor_sorted.cpu().numpy(), batch_first=self.params['batch_first'])
			packed_question_batch_encoding, question_batch_encoder_hidden = self.question_encoder(packed_question_batch_embedding, question_batch_encoder_hidden)
			if self.params['rnn_type'] == 'LSTM':
				question_batch_encoder_hidden = (question_batch_encoder_hidden[0].transpose(0,1), question_batch_encoder_hidden[1].transpose(0,1))
				question_batch_encoder_hidden = (question_batch_encoder_hidden[0][question_perm_idx_resort], question_batch_encoder_hidden[1][question_perm_idx_resort])
				question_batch_encoder_hidden = (question_batch_encoder_hidden[0].transpose(0,1), question_batch_encoder_hidden[1].transpose(0,1))
			else:
				question_batch_encoder_hidden = question_batch_encoder_hidden.transpose(0,1)
				question_batch_encoder_hidden = question_batch_encoder_hidden[question_perm_idx_resort]
				question_batch_encoder_hidden = question_batch_encoder_hidden.transpose(0,1)
		else:
			if not self.params['batch_first']:
				question_batch_embedding = question_batch_embedding.transpose(0,1)
			output_question_batch_encoding, question_batch_encoder_hidden = self.question_encoder(question_batch_embedding, question_batch_encoder_hidden)

		#Fact Encoder
		fact_batch_encoder_hidden = self.fact_encoder.initHidden(batch_size=fact_batch_embedding.size(0))

		if batch_mode:
			fact_batch_length_tensor_sorted, fact_perm_idx = fact_batch_len.sort(0, descending=True)
			_, fact_perm_idx_resort = fact_perm_idx.sort(0, descending=False)
			fact_batch_embedding = fact_batch_embedding[fact_perm_idx]

			if not self.params['batch_first']:
				fact_batch_embedding = fact_batch_embedding.transpose(0,1)
			packed_fact_batch_embedding = nn.utils.rnn.pack_padded_sequence(fact_batch_embedding, fact_batch_length_tensor_sorted.cpu().numpy(), batch_first=self.params['batch_first'])
			packed_fact_batch_encoding, fact_batch_encoder_hidden = self.fact_encoder(packed_fact_batch_embedding, fact_batch_encoder_hidden)
			if self.params['rnn_type'] == 'LSTM':
				fact_batch_encoder_hidden = (fact_batch_encoder_hidden[0].transpose(0,1), fact_batch_encoder_hidden[1].transpose(0,1))
				fact_batch_encoder_hidden = (fact_batch_encoder_hidden[0][fact_perm_idx_resort], fact_batch_encoder_hidden[1][fact_perm_idx_resort])
				fact_batch_encoder_hidden = (fact_batch_encoder_hidden[0].transpose(0,1), fact_batch_encoder_hidden[1].transpose(0,1))
			else:
				fact_batch_encoder_hidden = fact_batch_encoder_hidden.transpose(0,1)
				fact_batch_encoder_hidden = fact_batch_encoder_hidden[fact_perm_idx_resort]
				fact_batch_encoder_hidden = fact_batch_encoder_hidden.transpose(0,1)
		else:
			if not self.params['batch_first']:
				fact_batch_embedding = fact_batch_embedding.transpose(0,1)
			output_fact_batch_encoding, fact_batch_encoder_hidden = self.fact_encoder(fact_batch_embedding, fact_batch_encoder_hidden)


		#Attention
		if self.params['rnn_type'] == 'LSTM':
			question_batch_encoding = question_batch_encoder_hidden[0].narrow(0,self.params['num_layers']-1,1)
			fact_batch_encoding = fact_batch_encoder_hidden[0].narrow(0,self.params['num_layers']-1,1)
		else:
			question_batch_encoding = question_batch_encoder_hidden.narrow(0,self.params['num_layers']-1,1)
			fact_batch_encoding = fact_batch_encoder_hidden.narrow(0,self.params['num_layers']-1,1)


		#Image Encoder
		image_batch_encoding = self.image_encoder(image_batch).unsqueeze(0)

		#Combine Encodings
		combined_encoding_batch = torch.cat((question_batch_encoding, fact_batch_encoding, image_batch_encoding),2)
		# if self.params['rnn_type'] == 'LSTM':
		# 	combined_encoding_batch = torch.cat((question_batch_encoder_hidden[0].narrow(0,self.params['num_layers']-1,1), fact_batch_encoder_hidden[0].narrow(0,self.params['num_layers']-1,1), image_batch_encoding), 2)
		# else:
		# 	combined_encoding_batch = torch.cat((question_batch_encoder_hidden.narrow(0,self.params['num_layers']-1,1), fact_batch_encoder_hidden.narrow(0,self.params['num_layers']-1,1), image_batch_encoding), 2)

		#History Encoder
		combined_encoding_batch = combined_encoding_batch.view(-1, self.params['num_dialogs'], combined_encoding_batch.size(2))
		history_batch_encoder_hidden = self.history_encoder.initHidden(batch_size=combined_encoding_batch.size(0))
		if not self.params['batch_first']:
			combined_encoding_batch = combined_encoding_batch.transpose(0,1)
		final_history_encoding, history_batch_encoder_hidden = self.history_encoder(combined_encoding_batch, history_batch_encoder_hidden)
		if not self.params['batch_first']:
			final_history_encoding = final_history_encoding.transpose(0,1).contiguous()

		return final_history_encoding, question_batch_encoder_hidden
