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
        self.history_encoder = ABot.HistoryEncoder(params['hidden_dim'], params['hidden_dim'], num_layers=params['num_layers'], rnn=params['rnn_type'], batch_first=params['batch_first'])
        # self.attention_layer = ABot.Attention(params['hidden_dim'], params['hidden_dim'], params['hidden_dim'], params['filter_size'], params['image_dim'], params['image_feature'], params['image_embed_size'])
        self.history_attention = ABot.HistoryAttention(params['hidden_dim'], 1)
        self.linear = nn.Linear(params['hidden_dim']*2 + params['image_embed_size'], params['embedding_size'])

    def forward(self, question_batch_embedding, question_batch_len, fact_batch_embedding, fact_batch_len, image_batch, batch_mode=True, dropout=0.5):
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

		#Fact Encoder
        fact_batch_encoder_hidden = self.fact_encoder.initHidden(batch_size=fact_batch_embedding.size(0))
        if batch_mode:
            fact_batch_length_tensor_sorted, fact_perm_idx = fact_batch_len.sort(0, descending=True)
            _, fact_perm_idx_resort = fact_perm_idx.sort(0, descending=False)
            fact_batch_embedding = fact_batch_embedding.index_select(0,fact_perm_idx)
            if not self.params['batch_first']:
                fact_batch_embedding = fact_batch_embedding.transpose(0,1)
            packed_fact_batch_embedding = nn.utils.rnn.pack_padded_sequence(fact_batch_embedding, fact_batch_length_tensor_sorted.data.cpu().numpy(), batch_first=self.params['batch_first'])
            packed_fact_batch_encoding, fact_batch_encoder_hidden = self.fact_encoder(packed_fact_batch_embedding, fact_batch_encoder_hidden)
            output_fact_batch_encoding, _ = nn.utils.rnn.pad_packed_sequence(packed_fact_batch_encoding)
            output_fact_batch_encoding = output_fact_batch_encoding.index_select(sort_index, fact_perm_idx_resort)
            if self.params['rnn_type'] == 'LSTM':
                fact_batch_encoder_hidden = (fact_batch_encoder_hidden[0].index_select(1, fact_perm_idx_resort), fact_batch_encoder_hidden[1].index_select(1, fact_perm_idx_resort))
            else:
                fact_batch_encoder_hidden = fact_batch_encoder_hidden.index_select(1, fact_perm_idx_resort)
        else:
            if not self.params['batch_first']:
                fact_batch_embedding = fact_batch_embedding.transpose(0,1)
            output_fact_batch_encoding, fact_batch_encoder_hidden = self.fact_encoder(fact_batch_embedding, fact_batch_encoder_hidden)

        fact_batch_encoding = fact_batch_encoder_hidden[0].narrow(0, self.params['num_layers']-1,1)
        fact_batch_encoding = fact_batch_encoding.squeeze(0).view(batchSize,-1, self.params['hidden_dim'])
        attended_history = self.history_attention(fact_batch_encoding, question_batch_encoding, dropout=dropout)
        attended_history = attended_history.transpose(0,1)

		#Image Encoder
        image_batch_encoding = self.image_encoder(image_batch).unsqueeze(0)

        combined_embedding = torch.cat((question_batch_encoding.transpose(0,1), attended_history, image_batch_encoding), 2)
        combined_encoding_batch_linear = F.tanh(self.linear(F.dropout(combined_embedding, dropout, training=self.training)))
        combined_encoding_batch_linear = combined_encoding_batch_linear.transpose(0,1)
        return combined_encoding_batch_linear, question_batch_encoder_hidden
