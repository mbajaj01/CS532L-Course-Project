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
params['num_layers'] = 1
params['hidden_dim'] = 512
params['embed_size'] = 300
params['vocab_size'] = len(data.ind2word.keys())
params['embedding_size'] = 300
params['vgg_out'] = 4096
params['image_feature'] = 512
params['image_dim'] = 7
params['batch_size'] = 200
params['image_embed_size'] = 300
params['batch_size']=20
params['epochs'] = 20
params['rnn_type'] = 'LSTM'
params['num_dialogs'] = 10
params['sampleWords'] = False
params['temperature'] = 0.8
params['beamSize'] = 5
params['beamLen'] = 20
params['word2ind'] = data.word2ind
params['ind2word'] = data.ind2word
params['USE_CUDA'] = USE_CUDA
params['gpu'] = gpu
params['filter_size'] = 3
compute_ranks = False

#Define Models
AEncoder = ABot_Encoder.ABotEncoder(params)
ADecoder = ABot_Decoder.ABotDecoder(params)
embedding_weights = np.random.random((params['vocab_size'], params['embed_size']))
embedding_weights[0,:] = np.zeros((1, params['embed_size']))
print (embedding_weights)
embedding_layer = ABot.EmbeddingLayer(embedding_weights)

if compute_ranks:
	checkpoint = torch.load('../outputs/supervised_ABot_V2_LSTM_2Layers_test_10')
	print ("DONE")
	AEncoder.load_state_dict(checkpoint['AEncoder'])
	ADecoder.load_state_dict(checkpoint['ADecoder'])
	embedding_layer.load_state_dict(checkpoint['embedding_layer'])
#Criterion
criterion = nn.CrossEntropyLoss(reduce=False)

#Optimizer
optimizer = torch.optim.Adam([{'params':AEncoder.parameters()},{'params':ADecoder.parameters()},{'params':embedding_layer.parameters()}], lr=0.001)

if USE_CUDA:
	AEncoder = AEncoder.cuda(gpu)
	ADecoder = ADecoder.cuda(gpu)
	embedding_layer = embedding_layer.cuda(gpu)

def supervisedTrain(batch, AEncoder, ADecoder, embedding_layer, params, optimizer, criterion, onlyForward=False, num_dialogs=10, use_teacher_forcing=True, batch_mode=True, volatile=False, ranker=False):
	#Get Data
	optimizer.zero_grad()

	question_batch = batch['questions'].reshape((-1, batch['questions'].shape[2])).astype(np.int)
	answer_input_batch = batch['answers_input'].reshape((-1, batch['answers_input'].shape[2])).astype(np.int)
	answer_output_batch = batch['answers_output'].reshape((-1, batch['answers_output'].shape[2])).astype(np.int)
	question_batch_len = batch['questions_length'].reshape((-1,)).astype(np.int)
	answer_batch_len = batch['answers_length'].reshape((-1, )).astype(np.int)
	history_batch = batch['history'].reshape((-1, batch['history'].shape[2])).astype(np.int)
	history_batch_len = batch['history_length'].reshape((-1,)).astype(np.int)
	image_batch = batch['images'].astype(np.float)
	image_batch = image_batch.reshape((-1,1,image_batch.shape[1]))
	image_batch = np.repeat(image_batch, batch['questions'].shape[1], 1)
	image_batch = image_batch.reshape((-1, image_batch.shape[2]))

	if ranker:
		options_batch_len = batch['options_length'].reshape((-1, batch['options_length'].shape[2])).astype(np.int)
		options_input_batch = batch['options_input'].reshape((-1, batch['options_input'].shape[2], batch['options_input'].shape[3])).astype(np.int)
		options_output_batch = batch['options_output'].reshape((-1, batch['options_input'].shape[2], batch['options_output'].shape[3])).astype(np.int)
		answer_ground_truth = batch['answers_indexes'].reshape((-1,)).astype(np.int)

	#Create Tensors
	question_batch_tensor = Variable(torch.from_numpy(question_batch).long(), volatile=volatile)
	answer_input_batch_tensor = Variable(torch.from_numpy(answer_input_batch).long(), volatile=volatile)
	answer_output_batch_tensor = Variable(torch.from_numpy(answer_output_batch).long(), volatile=volatile)
	history_batch_tensor = Variable(torch.from_numpy(history_batch).long(), volatile=volatile)
	image_batch_tensor = Variable(torch.from_numpy(image_batch).float(), volatile=volatile)
	question_batch_length_tensor = torch.from_numpy(question_batch_len).long()
	history_batch_length_tensor = torch.from_numpy(history_batch_len).long()
	answer_batch_length_tensor = torch.from_numpy(answer_batch_len).long()

	if ranker:
		options_input_batch_tensor = Variable(torch.from_numpy(options_input_batch).long(), volatile=volatile)
		options_output_batch_tensor = Variable(torch.from_numpy(options_output_batch).long(), volatile=volatile)
		options_batch_length_tensor = torch.from_numpy(options_batch_len).long()
		answer_ground_truth_tensor = Variable(torch.from_numpy(answer_ground_truth).long(), volatile=volatile)

	if USE_CUDA:
		question_batch_tensor = question_batch_tensor.cuda(gpu)
		answer_input_batch_tensor = answer_input_batch_tensor.cuda(gpu)
		answer_output_batch_tensor = answer_output_batch_tensor.cuda(gpu)
		history_batch_tensor = history_batch_tensor.cuda(gpu)
		image_batch_tensor = image_batch_tensor.cuda(gpu)
		question_batch_length_tensor = question_batch_length_tensor.cuda(gpu)
		history_batch_length_tensor = history_batch_length_tensor.cuda(gpu)
		if ranker:
			options_input_batch_tensor = options_input_batch_tensor.cuda(gpu)
			options_output_batch_tensor = options_output_batch_tensor.cuda(gpu)
			answer_ground_truth_tensor = answer_ground_truth_tensor.cuda(gpu)

	question_batch_embedding = embedding_layer(question_batch_tensor)
	fact_batch_embedding = embedding_layer(history_batch_tensor)

	final_history_encoding, question_batch_encoder_hidden = AEncoder(question_batch_embedding, question_batch_length_tensor, fact_batch_embedding, history_batch_length_tensor, image_batch_tensor, batch_mode=batch_mode)
	final_history_encoding = final_history_encoding.view(-1, final_history_encoding.size(2)).unsqueeze(0)

	answer_input_batch_embedding = embedding_layer(answer_input_batch_tensor)
	if params['num_layers'] > 1:
		if params['rnn_type'] == 'LSTM':
			answer_decoder_batch_hidden = (torch.cat((question_batch_encoder_hidden[0].narrow(0,0,params['num_layers'] - 1), final_history_encoding), 0), question_batch_encoder_hidden[1])
		else:
			answer_decoder_batch_hidden = torch.cat((question_batch_encoder_hidden.narrow(0,0,params['num_layers'] - 1), final_history_encoding), 0)
	else:
		if params['rnn_type'] == 'LSTM':
			answer_decoder_batch_hidden = (final_history_encoding, question_batch_encoder_hidden[1])
		else:
			answer_decoder_batch_hidden = final_history_encoding

	if ranker:
		options_input_batch_tensor = options_input_batch_tensor.transpose(0,1)
		options_output_batch_tensor = options_output_batch_tensor.transpose(0,1)
		options_batch_length_tensor = options_batch_length_tensor.transpose(0,1)
		sentence_probabilities = Variable(torch.zeros(batch['options_input'].shape[2], batch['options_input'].shape[0]*batch['options_input'].shape[1]), volatile=volatile).float()
		if USE_CUDA:
			sentence_probabilities = sentence_probabilities.cuda(gpu)

		for options_num in range(batch['options_input'].shape[2]):
			current_options_length_tensor = options_batch_length_tensor[options_num, :]
			current_options_input_tensor = options_input_batch_tensor[options_num, :, :]
			current_options_output_tensor = options_output_batch_tensor[options_num, :, :]

			current_options_input_embeddings = embedding_layer(current_options_input_tensor)
			if params['rnn_type'] == 'LSTM':
				current_probabilities_option_batch, _ = ADecoder(current_options_input_embeddings, (answer_decoder_batch_hidden[0].clone(), answer_decoder_batch_hidden[1].clone()))
			else:
				current_probabilities_option_batch, _ = ADecoder(current_options_input_embeddings, answer_decoder_batch_hidden.clone())

			sentence_probabilities[options_num, :] =  computeProbability(current_options_output_tensor, current_probabilities_option_batch)

		sentence_probabilities = sentence_probabilities.transpose(0,1)
		ranks = computeRanks(sentence_probabilities, answer_ground_truth_tensor)
		return ranks.sum(0).data.cpu().numpy(), ranks.data.cpu().size(0)
	else:
		probabilities_answer_batch, answer_decoder_batch_hidden = ADecoder(answer_input_batch_embedding, answer_decoder_batch_hidden, batch_mode=batch_mode)
		probabilities_answer_batch = probabilities_answer_batch.view(-1, probabilities_answer_batch.size(2))
		answer_output_batch_tensor = answer_output_batch_tensor.view(-1,)
		loss = criterion(probabilities_answer_batch, answer_output_batch_tensor)
		mask = answer_output_batch_tensor != 0
		mask = mask.float()
		loss_masked = torch.mul(loss, mask)
		loss_masked = loss_masked.sum()
		if not onlyForward:
			loss_masked.backward()
			parameters = list(AEncoder.parameters())+list(ADecoder.parameters())+list(embedding_layer.parameters())
			parameters = list(filter(lambda p: p.grad is not None, parameters))
			# print ("Number of Params:", len(params))
			for p in parameters:
				p.grad.data.clamp_(-5.0, 5.0)
			optimizer.step()
		#loss_masked = loss_masked/ mask.sum()
		return loss_masked.data.cpu().numpy(), mask.sum().data.cpu().numpy()

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

def train(data, params, AEncoder, ADecoder, embedding_layer, criterion, optimizer):
	number_of_batches = math.ceil(data.datasize['train']/params['batch_size'])
	number_of_val_batches = math.ceil(data.datasize['val']/params['batch_size'])
	indexes = np.arange(data.datasize['train'])
	val_indexes = np.arange(data.datasize['val'])

	print (number_of_batches, number_of_val_batches)
	lr = 0.002
	lr_decay = 0.9997592083
	#lr_decay = 1.0
	min_lr = 5e-5
	avgloss = 0.0
	tokens = 0.0
	for epoch in range(params['epochs']):
		np.random.shuffle(indexes)
		start = time.time()
		avg_loss_arr = []
		loss_arr = []
		avgvalloss = 0.0
		val_tokens = 0.0
		for batch_num in range(number_of_batches):
			batch_indexes = indexes[batch_num*params['batch_size']:(batch_num+1)*params['batch_size']]
			#batch_indexes = np.random.randint(data.datasize['train'], size=params['batch_size'])
			#if batch_num == 0:
			#batch_indexes = np.array([ 49770,33486,40377,11948,9577,20869,16765,6823,30332,27758,1753,33905,41981,29280,22954,36208,40717,4615,39403,40655]) - 1
			#else:
			#	batch_indexes = np.array([ 29173,41365,36417,36532,17365,20208,44022,21525,12940,25041,32303,34575,25448,166, 35768,41220,15860,26001,38355, 49128]) - 1
			batch = data.getBatch(batch_indexes, 'train')
			loss = supervisedTrain(batch, AEncoder, ADecoder, embedding_layer, params, optimizer, criterion, use_teacher_forcing=True, batch_mode=True)
			#avgloss += loss[0][0]
			tokens = loss[1][0]
			if avgloss > 0:
				avgloss = 0.95 * avgloss + 0.05 * (loss[0][0]/tokens)
			else:
				avgloss = loss[0][0]/tokens
			loss_arr.append(loss)
			#if lr > min_lr:
				#lr *= lr_decay
				#for param_group in optimizer.param_groups:
				#	param_group['lr'] = lr
			if batch_num%100 == 0:
				#print ("Done Batch:", batch_num, "\tAverage Loss Per Batch:", avgloss/(batch_num+1), "\t Current Batch Loss: ", loss, "\tlr:",lr)
				print ("Done Batch:", batch_num, "\tAverage Loss Per Batch:", avgloss, "\t Current Batch Loss: ", loss[0][0]/tokens, "\tlr:",lr)

		for batch_val_num in range(number_of_val_batches):
			batch_val_indexes = val_indexes[batch_val_num*params['batch_size']:(batch_val_num+1)*params['batch_size']]
			batch_val = data.getBatch(batch_val_indexes, 'val')
			val_loss = supervisedTrain(batch, AEncoder, ADecoder, embedding_layer, optimizer, criterion, use_teacher_forcing=True, onlyForward=True)
			val_tokens += val_loss[1][0]
			avgvalloss += val_loss[0][0]
		print ("Epoch:",epoch, "\tTime:", time.time() - start, "\tAverage Loss Per Batch::", avgloss, "\tAverage Validation Loss:", avgvalloss/val_tokens)
		#torch.save({'epoch': epoch ,'image_encoder': image_encoder.state_dict(),'embedding_layer': embedding_layer.state_dict(), 'question_encoder':question_encoder.state_dict(), 'fact_encoder':fact_encoder.state_dict(), 'history_encoder':history_encoder.state_dict(), 'answer_decoder':answer_decoder.state_dict(), 'optimizer':optimizer.state_dict()}, "outputs/supervised_ABot_"+str(epoch))
		torch.save({'epoch': epoch ,'AEncoder': AEncoder.state_dict(),'ADecoder': ADecoder.state_dict(), 'embedding_layer':embedding_layer.state_dict(), 'optimizer':optimizer.state_dict()}, "outputs/supervised_ABot_V2_LSTM_2Layers_"+str(epoch))

		loss_arr = np.array(loss_arr)
		np.save(open('outputs/loss_supervised_ABot_'+str(epoch), 'wb+'), loss_arr)


def getRanks(data, dataset, params, AEncoder, ADecoder, embedding_layer, criterion, optimizer):
	number_of_batches = math.ceil(data.datasize[dataset]/params['batch_size'])
	indexes = np.arange(data.datasize[dataset])
	print (number_of_batches)
	avgRank = 0.0
	elements = 0
	for batch_num in range(number_of_batches):
		batch_indexes = indexes[batch_num*params['batch_size']:(batch_num+1)*params['batch_size']]
		batch = data.getBatch(batch_indexes, 'test')
		ranks, currElems = supervisedTrain(batch, AEncoder, ADecoder, embedding_layer, params, optimizer, criterion, onlyForward=True, num_dialogs=10, use_teacher_forcing=True, batch_mode=True, volatile=True, ranker=True)
		avgRank += np.sum(ranks,0)
		elements += currElems
		if batch_num%5 == 0:
			print ("Done:", batch_num, "Average Rank", avgRank/elements)
	avgRank = avgRank/elements
	print ("Average Rank:", avgRank)

train(data, params, AEncoder, ADecoder, embedding_layer, criterion, optimizer)
#getRanks(data, 'test',  params, AEncoder, ADecoder, embedding_layer, criterion, optimizer)
