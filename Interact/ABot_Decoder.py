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

class ABotDecoder(nn.Module):
    def __init__(self, params):
        super(ABotDecoder, self).__init__()
        self.params = params
        self.answer_decoder = ABot.AnswerDecoder(self.params['embed_size'], self.params['hidden_dim'], self.params['vocab_size'], num_layers=self.params['num_layers'], rnn=self.params['rnn_type'], batch_first=self.params['batch_first'])
        # self.linear = nn.Linear( self.params['hidden_dim'], self.params['embed_size'])
        self.attention = ABot.DecoderAttention( self.params['hidden_dim'], self.params['embed_size'])

    def forward(self, answer_input_embedding, answer_decoder_batch_hidden, batch_mode=True):
        if not self.params['batch_first']:
            answer_input_embedding = answer_input_embedding.transpose(0,1)
        probabilities_answer_batch, answer_decoder_batch_hidden = self.answer_decoder(answer_input_embedding, answer_decoder_batch_hidden, isPacked=False)
        if not self.params['batch_first']:
            probabilities_answer_batch = probabilities_answer_batch.transpose(0,1).contiguous()
        return probabilities_answer_batch, answer_decoder_batch_hidden

    def beamSearch(self, input_tokens, answer_decoder_batch_hidden, embedding_layer, sequence_length=None):
        batch_size = input_tokens.size(0)
        # answer_decoder_batch_hidden = question_hidden
        if sequence_length is None:
            sequence_length = self.params['beamLen']

        beam_size = self.params['beamSize']
        # if self.params['rnn_type'] == 'LSTM':
        #     answer_decoder_batch_hidden[0][self.params['num_layers'] - 1] = encoder_output
        # else:
        #     answer_decoder_batch_hidden[self.params['num_layers'] - 1] = encoder_output
        # input_tokens = answer_input_tokens.narrow(1,0,1)
        input_tokens_embedding = embedding_layer(input_tokens)
        if not self.params['batch_first']:
            input_tokens_embedding = input_tokens_embedding.transpose(0,1)
        output_probabilities, answer_decoder_batch_hidden = self.answer_decoder(input_tokens_embedding, answer_decoder_batch_hidden, isPacked=False)
        if self.params['rnn_type'] == 'LSTM':
            answer_decoder_batch_hidden = (answer_decoder_batch_hidden[0].view(-1,1,self.params['hidden_dim']), answer_decoder_batch_hidden[1].view(-1,1,self.params['hidden_dim']))
            beam_hidden_state = (answer_decoder_batch_hidden[0].repeat(1,beam_size,1).view(self.params['num_layers'], -1, self.params['hidden_dim']), answer_decoder_batch_hidden[1].repeat(1,beam_size,1).view(self.params['num_layers'], -1, self.params['hidden_dim']))
        else:
            answer_decoder_batch_hidden = answer_decoder_batch_hidden.view(-1,1,self.params['hidden_dim'])
            beam_hidden_state = answer_decoder_batch_hidden.repeat(1,beam_size,1).view(self.params['num_layers'], -1, self.params['hidden_dim'])
        output_probabilities = F.log_softmax(output_probabilities, 2)
        values, indices = torch.topk(output_probabilities, beam_size, dim=2, largest=True, sorted=False)
        sequence_all = Variable(torch.zeros(batch_size, beam_size, sequence_length).long())
        sequence = Variable(torch.zeros(sequence_length, batch_size).long())
        sequence_probabilities = Variable(torch.zeros(sequence_length, batch_size, beam_size).float())
        EOS_TOKEN = self.params['word2ind']['<END>']
        masked_vector = Variable(torch.zeros(1, self.params['vocab_size']).float())
        masked_vector = masked_vector - 99999
        masked_vector[0,0] = 0

        indexer = Variable(torch.arange(batch_size).long().unsqueeze(1).expand_as(indices.squeeze(0))*beam_size)
        masking_batch_num = Variable(torch.arange(batch_size*beam_size).long())
        if self.params['USE_CUDA']:
            sequence_all = sequence_all.cuda(self.params['gpu'])
            sequence = sequence.cuda(self.params['gpu'])
            sequence_probabilities = sequence_probabilities.cuda(self.params['gpu'])
            indexer = indexer.cuda(self.params['gpu'])
            masking_batch_num = masking_batch_num.cuda(self.params['gpu'])
            masked_vector = masked_vector.cuda(self.params['gpu'])

        sequence_all[:,:,0] = indices
        sequence_probabilities[0] = values
        beam_probability_sum = values.clone().squeeze(0).unsqueeze(2)
        for current_index in range(1, sequence_length):
            #Select next words
            current_input_words = sequence_all[:,:,current_index - 1].clone().view(-1,1)
            mask = sequence_all == EOS_TOKEN
            mask = torch.max(mask, 2)[0].view(-1,)
            lengths = (sequence_all != 0).float()
            lengths = torch.sum(lengths, 2).unsqueeze(2)
            current_input_words_embeddings = embedding_layer(current_input_words)
            if not self.params['batch_first']:
                current_input_words_embeddings = current_input_words_embeddings.transpose(0,1)

			#Pass through Decoder
            current_output_probabilities, beam_hidden_state = self.answer_decoder(current_input_words_embeddings, beam_hidden_state, isPacked=False)
            current_output_probabilities = F.log_softmax(current_output_probabilities, 2).view(-1, current_output_probabilities.size(2))

			#Masking EOS
            masked_indexes = masking_batch_num[mask]
            if len(masked_indexes.size()) > 0:
                masking_vectors = masked_vector.repeat(masked_indexes.size(0), 1)
                current_output_probabilities.index_copy_(0, masked_indexes, masking_vectors)

            current_output_probabilities = current_output_probabilities.view(batch_size, beam_size, -1)
            # lengths = lengths.expand_as(current_output_probabilities)

            #Update Parameters for next iteration
            current_total_probabilities = beam_probability_sum.expand_as(current_output_probabilities) + current_output_probabilities
            # current_total_probabilities = current_total_probabilities/lengths
            current_values, current_indices = torch.topk(current_total_probabilities.view(batch_size,-1), beam_size, dim=1, largest=True, sorted=False)
            next_indices = current_indices/self.params['vocab_size']
            next_words = current_indices%self.params['vocab_size']

            next_indices_adjusted = next_indices + indexer
            next_indices_adjusted = next_indices_adjusted.view(-1,)

            sequence_all = torch.index_select(sequence_all.view(-1, sequence_length), 0, next_indices_adjusted).view(-1, beam_size, sequence_length)
            sequence_all[:,:,current_index] = next_words

            if self.params['rnn_type'] == 'LSTM':
                beam_hidden_state = (beam_hidden_state[0].transpose(0,1), beam_hidden_state[1].transpose(0,1))
                next_beam_hidden_state = (torch.index_select(beam_hidden_state[0], 0, next_indices_adjusted), torch.index_select(beam_hidden_state[1], 0, next_indices_adjusted))
                next_beam_hidden_state = (next_beam_hidden_state[0].transpose(0,1), next_beam_hidden_state[1].transpose(0,1))
            else:
                beam_hidden_state = beam_hidden_state.transpose(0,1)
                next_beam_hidden_state = torch.index_select(beam_hidden_state, 0, next_indices_adjusted)
                next_beam_hidden_state = next_beam_hidden_state.transpose(0,1)

            beam_probability_sum = current_values.unsqueeze(2)
            beam_hidden_state = next_beam_hidden_state

        return sequence_all, beam_probability_sum
        # test_index = 0
        # # print (lengths)
        # # print (beam_probability_sum[test_index])
        # tokens = sequence_all.data.cpu().numpy()
        # for tok in tokens[test_index]:
        #     print (" ".join([self.params['ind2word'][x] for x in tok if x != 0]))
        #
        # # tokens = answer_input_tokens.data.cpu().numpy()
        # # print (" ".join([self.params['ind2word'][x] for x in tokens[test_index] if x != 0]))
        # # aa
