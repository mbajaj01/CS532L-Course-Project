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
import QBot

class QBotEncoder(nn.Module):
    def __init__(self, params):
        super(QBotEncoder, self).__init__()
        self.params = params
        self.answer_encoder = QBot.QuestionEncoder(params['embedding_size'], params['hidden_dim'], num_layers=params['num_layers'], rnn=params['rnn_type'], batch_first=params['batch_first'])
        self.fact_encoder = QBot.FactEncoder(params['embedding_size'], params['hidden_dim'], num_layers=params['num_layers'], rnn=params['rnn_type'], batch_first=params['batch_first'])
        self.history_attention = QBot.HistoryAttention(params['hidden_dim'], 1)
        self.linear = nn.Linear(params['hidden_dim']*2, params['embedding_size'])

    def forward(self, answer_batch_embedding, answer_batch_len, fact_batch_embedding, fact_batch_len, batch_mode=True, dropout=0.5):
        #batch_mode = False
        batchSize = answer_batch_embedding.size(0)
        sort_index = 0 if self.params['batch_first'] else 1

		#Answer Encoder
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
        attended_history = self.history_attention(fact_batch_encoding, answer_batch_encoding, dropout=dropout)
        attended_history = attended_history.transpose(0,1)

        combined_embedding = torch.cat((answer_batch_encoding.transpose(0,1), attended_history), 2)
		#Image Encoder
        combined_encoding_batch_linear = F.tanh(self.linear(F.dropout(combined_embedding, dropout, training=self.training)))
        combined_encoding_batch_linear = combined_encoding_batch_linear.transpose(0,1)
        return combined_encoding_batch_linear, answer_batch_encoder_hidden, combined_embedding
