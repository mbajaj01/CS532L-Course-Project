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
import json
import h5py
import time
import dataloader
import math
import QBot_Encoder
import QBot_Decoder

torch.backends.cudnn.enabled = False
random.seed(32)
np.random.seed(32)
torch.manual_seed(7)
torch.cuda.manual_seed_all(7)
#Load Data
dialog_loc = '../../../chat_processed_data.h5'
param_loc = '../../../chat_processed_params.json'
image_loc = '../../../data_img.h5'

data = dataloader.DataLoader(dialog_loc, image_loc, param_loc)
print ("Done: Data Preparation")

#CUDA
USE_CUDA = True
gpu = 0

#Parameters
params = {}
params['batch_first'] = False
params['num_layers'] = 2
params['hidden_dim'] = 512
params['embed_size'] = 300
params['vocab_size'] = len(data.ind2word.keys())
params['embedding_size'] = 300
params['vgg_out'] = 4096
params['batch_size'] = 200
params['image_embed_size'] = 300
params['batch_size']=128
params['epochs'] = 40
params['rnn_type'] = 'LSTM'
params['num_dialogs'] = 10
params['sampleWords'] = False
params['temperature'] = 0.3
params['beamSize'] = 5
params['beamLen'] = 20
params['word2ind'] = data.word2ind
params['ind2word'] = data.ind2word
params['USE_CUDA'] = USE_CUDA
params['gpu'] = gpu

compute_ranks = False
current_epoch = 15

#Define Models
QEncoder = QBot_Encoder.QBotEncoder(params)
QDecoder = QBot_Decoder.QBotDecoder(params)
embedding_weights = np.random.random((params['vocab_size'], params['embed_size']))
embedding_weights[0,:] = np.zeros((1, params['embed_size']))
print (embedding_weights)
embedding_layer = QBot.EmbeddingLayer(embedding_weights)
sampler = QBot.GumbelSampler()


#Criterion
criterion = {}
criterion['CrossEntropyLoss'] = nn.CrossEntropyLoss(reduce=False)
criterion['HingeEmbeddingLoss'] = nn.HingeEmbeddingLoss(margin=0.0, size_average=False)

#Optimizer
optimizer = torch.optim.Adam([{'params':QEncoder.parameters()},{'params':QDecoder.parameters()},{'params':embedding_layer.parameters()}], lr=0.001)

#Load Models
if current_epoch > 0:
	checkpoint = torch.load('outputs/supervised_QBot_HistoryAttention_2Layers_'+str(current_epoch-1))
	QEncoder.load_state_dict(checkpoint['AEncoder'])
	QDecoder.load_state_dict(checkpoint['ADecoder'])
	embedding_layer.load_state_dict(checkpoint['embedding_layer'])
	optimizer.load_state_dict(checkpoint['optimizer'])

if USE_CUDA:
	QEncoder = QEncoder.cuda(gpu)
	QDecoder = QDecoder.cuda(gpu)
	embedding_layer = embedding_layer.cuda(gpu)
	sampler = sampler.cuda(gpu)

QEncoder.eval()
QDecoder.eval()
embedding_layer.eval()

def processOptions(batch, QEncoder, QDecoder, embedding_layer, params, onlyForward=False, num_dialogs=10, use_teacher_forcing=True, batch_mode=True, volatile=True):
	average_loss = 0.0
	token_count = 0.0
	total_rank = 0.0
	total_tokens = 0.0
	for dialog_num in range(num_dialogs):
        #Get Data
		answer_batch = batch['answers'][:,dialog_num,:].astype(np.int)
		question_input_batch = batch['questions_input'][:,dialog_num,:].astype(np.int)
		question_output_batch = batch['questions_output'][:,dialog_num,:].astype(np.int)
		question_batch_len = batch['questions_length'][:,dialog_num].astype(np.int)
		answer_batch_len = batch['answers_length'][:,dialog_num].astype(np.int)
		history_batch = batch['history'][:,:dialog_num+1,:].astype(np.int)
		history_batch_len = batch['history_length'][:,:dialog_num+1].astype(np.int)
		image_batch = batch['images'].astype(np.float)


		#Create Tensors
		answer_batch_tensor = Variable(torch.from_numpy(answer_batch).long(), volatile=volatile)
		question_input_batch_tensor = Variable(torch.from_numpy(question_input_batch).long(), volatile=volatile)
		question_output_batch_tensor = Variable(torch.from_numpy(question_output_batch).long(), volatile=volatile)
		history_batch_tensor = Variable(torch.from_numpy(history_batch).long(), volatile=volatile)
		image_batch_tensor = Variable(torch.from_numpy(image_batch).float(), volatile=volatile)
		question_batch_length_tensor = Variable(torch.from_numpy(question_batch_len).long(), volatile=volatile)
		history_batch_length_tensor = Variable(torch.from_numpy(history_batch_len).long(), volatile=volatile)
		answer_batch_length_tensor = Variable(torch.from_numpy(answer_batch_len).long(), volatile=volatile)
		#noise_input = Variable(torch.FloatTensor(params['beamLen'], batch_size, params['vocab_size']).uniform_(0,1), volatile=volatile)

		if USE_CUDA:
			answer_batch_tensor = answer_batch_tensor.cuda(gpu)
			question_input_batch_tensor = question_input_batch_tensor.cuda(gpu)
			question_output_batch_tensor = question_output_batch_tensor.cuda(gpu)
			history_batch_tensor = history_batch_tensor.cuda(gpu)
			image_batch_tensor = image_batch_tensor.cuda(gpu)
			question_batch_length_tensor = question_batch_length_tensor.cuda(gpu)
			history_batch_length_tensor = history_batch_length_tensor.cuda(gpu)
			answer_batch_length_tensor = answer_batch_length_tensor.cuda(gpu)
			#noise_input = noise_input.cuda(gpu)

		answer_batch_embedding = embedding_layer(answer_batch_tensor)
		fact_batch_embedding = embedding_layer(history_batch_tensor.view(-1, history_batch_tensor.size(2)))
		history_batch_length_tensor = history_batch_length_tensor.view(-1,)
		# fact_batch_embedding = fact_batch_embedding.view(-1, dialog_num+1, fact_batch_embedding.size(1), fact_batch_embedding.size(2))

		final_history_encoding, question_batch_encoder_hidden, combined_embedding = QEncoder(answer_batch_embedding, answer_batch_length_tensor, fact_batch_embedding, history_batch_length_tensor, batch_mode=batch_mode)
		_ , question_decoder_batch_hidden, _ = QDecoder(final_history_encoding, question_batch_encoder_hidden, batch_mode=batch_mode)
		QDecoder.beamSearch(question_input_batch_tensor, question_decoder_batch_hidden, embedding_layer)
		aaoptions_input_batch_tensor = options_input_batch_tensor.transpose(0,1)
		options_output_batch_tensor = options_output_batch_tensor.transpose(0,1)
		options_batch_length_tensor = options_batch_length_tensor.transpose(0,1)
		sentence_probabilities = Variable(torch.zeros(batch['options_input'].shape[2], batch['options_input'].shape[0]), volatile=volatile).float()
		if USE_CUDA:
			sentence_probabilities = sentence_probabilities.cuda(gpu)

		for options_num in range(batch['options_input'].shape[2]):
			current_options_length_tensor = options_batch_length_tensor[options_num]
			current_options_input_tensor = options_input_batch_tensor[options_num]
			current_options_output_tensor = options_output_batch_tensor[options_num]

			current_options_input_embeddings = embedding_layer(current_options_input_tensor)
			if params['rnn_type'] == 'LSTM':
				current_probabilities_option_batch, _, _ = ADecoder(current_options_input_embeddings, (answer_decoder_batch_hidden[0].clone(), answer_decoder_batch_hidden[1].clone()))
			else:
				current_probabilities_option_batch, _, _ = ADecoder(current_options_input_embeddings, answer_decoder_batch_hidden.clone())

			sentence_probabilities[options_num] =  computeProbability(current_options_output_tensor, current_probabilities_option_batch)

		sentence_probabilities = sentence_probabilities.transpose(0,1)
		ranks = computeRanks(sentence_probabilities, answer_ground_truth_tensor)
		total_rank += ranks.sum(0).data.cpu().numpy()[0]
		total_tokens += ranks.size(0)
	return total_rank, total_tokens

def computeProbability(trueOutput, probabilities):
	mask = trueOutput != 0
	mask = mask.float()
	logProbs = F.log_softmax(probabilities, 2)
	sentenceProbs = torch.gather(logProbs, 2, trueOutput.unsqueeze(2)).squeeze(2)
	sentenceProbs = torch.mul(sentenceProbs, mask)
	sentenceProbs = sentenceProbs.sum(1)
	return sentenceProbs

def computeRanks(probabilities, ground_truth):
	scores = torch.gather(probabilities, 1, ground_truth.unsqueeze(1))
	ranks = probabilities > scores.repeat(1,probabilities.size(1))
	ranks = ranks.float()
	ranks = ranks.sum(1) + 1
	return ranks


def getRanks(data, dataset, params, AEncoder, ADecoder, embedding_layer, criterion, optimizer):
	number_of_batches = math.ceil(data.datasize[dataset]/params['batch_size'])
	indexes = np.arange(data.datasize[dataset])
	print (number_of_batches)
	avgRank = 0.0
	elements = 0
	for batch_num in range(number_of_batches):
		batch_indexes = indexes[batch_num*params['batch_size']:(batch_num+1)*params['batch_size']]
		batch = data.getQBatch(batch_indexes, 'test')
		ranks, currElems = processOptions(batch, QEncoder, QDecoder, embedding_layer, params, onlyForward=True, num_dialogs=10, use_teacher_forcing=True, batch_mode=True, volatile=True)
		avgRank += ranks
		elements += currElems
		if batch_num%5 == 0:
			print ("Done:", batch_num, "Average Rank", avgRank/elements)
	avgRank = avgRank/elements
	print ("Average Rank:", avgRank)

getRanks(data, 'test',  params, QEncoder, QDecoder, embedding_layer, criterion, optimizer)
