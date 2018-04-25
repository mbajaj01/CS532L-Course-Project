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
USE_CUDA = False
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
params['epochs'] = 8
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
current_epoch_ABot = 26
current_epoch_QBot = 23

#Define Models
AEncoder = ABot_Encoder.ABotEncoder(params)
ADecoder = ABot_Decoder.ABotDecoder(params)
QEncoder = QBot_Encoder.QBotEncoder(params)
QDecoder = QBot_Decoder.QBotDecoder(params)
embedding_weights = np.random.random((params['vocab_size'], params['embed_size']))
embedding_weights[0,:] = np.zeros((1, params['embed_size']))
print (embedding_weights)
ABot_embedding_layer = ABot.EmbeddingLayer(embedding_weights)
QBot_embedding_layer = QBot.EmbeddingLayer(embedding_weights)
sampler = ABot.GumbelSampler()


#Criterion
criterion = {}
criterion['CrossEntropyLoss'] = nn.CrossEntropyLoss(reduce=False)
criterion['HingeEmbeddingLoss'] = nn.HingeEmbeddingLoss(margin=0.0, size_average=False)
criterion['MSELoss'] = nn.MSELoss(size_average=False)

#Optimizer
ABot_optimizer = torch.optim.Adam([{'params':AEncoder.parameters()},{'params':ADecoder.parameters()},{'params':ABot_embedding_layer.parameters()}], lr=0.001)
QBot_optimizer = torch.optim.Adam([{'params':QEncoder.parameters()},{'params':QDecoder.parameters()},{'params':QBot_embedding_layer.parameters()}], lr=0.001)

#Load Models
if current_epoch_ABot > 0:
    #Load ABot
	print ("Loading")
	checkpoint = torch.load('outputs/supervised_ABot_HistoryAttention_AttentionRankingLoss_2Layers_'+str(current_epoch_ABot-1), map_location=lambda storage, loc: storage)
	AEncoder.load_state_dict(checkpoint['AEncoder'])
	ADecoder.load_state_dict(checkpoint['ADecoder'])
	ABot_embedding_layer.load_state_dict(checkpoint['embedding_layer'])
	ABot_optimizer.load_state_dict(checkpoint['optimizer'])


if current_epoch_QBot > 0:
    #Load QBot
	print ("Loading")
	checkpoint = torch.load('outputs/supervised_QBot_HistoryAttention_2Layers_Final_'+str(current_epoch_QBot-1), map_location=lambda storage, loc: storage)
	QEncoder.load_state_dict(checkpoint['AEncoder'])
	QDecoder.load_state_dict(checkpoint['ADecoder'])
	QBot_embedding_layer.load_state_dict(checkpoint['embedding_layer'])
	QBot_optimizer.load_state_dict(checkpoint['optimizer'])

AEncoder = AEncoder.cpu()
ADecoder = ADecoder.cpu()
QEncoder = QEncoder.cpu()
QDecoder = QDecoder.cpu()
ABot_embedding_layer = ABot_embedding_layer.cpu()
QBot_embedding_layer = QBot_embedding_layer.cpu()
if USE_CUDA:
	AEncoder = AEncoder.cuda(gpu)
	ADecoder = ADecoder.cuda(gpu)
	QEncoder = QEncoder.cuda(gpu)
	QDecoder = QDecoder.cuda(gpu)
	ABot_embedding_layer = ABot_embedding_layer.cuda(gpu)
	QBot_embedding_layer = QBot_embedding_layer.cuda(gpu)
	sampler = sampler.cuda(gpu)

def train(data, params, AEncoder, ADecoder, QEncoder, QDecoder, ABot_embedding_layer, QBot_embedding_layer, sampler, criterion, ABot_optimizer, QBot_optimizer, current_epoch=0):
	number_of_batches = math.ceil(data.datasize['test']/params['batch_size'])
	number_of_val_batches = math.ceil(data.datasize['val']/params['batch_size'])
	indexes = np.arange(data.datasize['test'])
	val_indexes = np.arange(data.datasize['val'])

	print (number_of_batches, number_of_val_batches)
	np.random.seed(1234)
	lr = 0.002
	lr_decay = 0.9997592083
	min_lr = 5e-5

	AEncoder.train()
	ADecoder.train()
	QEncoder.train()
	QDecoder.train()
	ABot_embedding_layer.train()
	QBot_embedding_layer.train()

	# print("Current Epoch:",current_epoch)
	for epoch in range(params['epochs']):
		numSL = 8 - epoch
		numSL = 0
		np.random.shuffle(indexes)
		start = time.time()
		avg_loss_arr = []
		loss_arr = []
		avgvalloss = 0.0
		val_tokens = 0.0
		avgloss = 0.0
		tokens = 0.0
		for batch_num in range(number_of_batches):
			batch_indexes = indexes[batch_num*params['batch_size']:(batch_num+1)*params['batch_size']]
			ABatch = data.getBatch(batch_indexes, 'test')
			QBatch = data.getQBatch(batch_indexes, 'test')
			loss = InteractTrain(ABatch, QBatch, AEncoder, ADecoder, QEncoder, QDecoder, ABot_embedding_layer, QBot_embedding_layer, sampler, params, ABot_optimizer, QBot_optimizer, criterion, numSL, batch_mode=True, volatile=True, onlyForward=True)
			tokens += loss[1]
			avgloss += loss[0]
			loss_arr.append(loss[0]/loss[1])
			#if lr > min_lr:
				#lr *= lr_decay
				#for param_group in optimizer.param_groups:
				#	param_group['lr'] = lr
			if batch_num%20 == 0:
				print ("Done Batch:", batch_num, "\tTime:",time.time() - start,"\tAverage Loss Per Batch:", avgloss/tokens, "\t Current Batch Loss: ", loss[0]/loss[1], "\tlr:",lr)

		for batch_val_num in range(number_of_val_batches):
			batch_val_indexes = val_indexes[batch_val_num*params['batch_size']:(batch_val_num+1)*params['batch_size']]
			ABatch_val = data.getBatch(batch_val_indexes, 'val')
			QBatch_val = data.getQBatch(batch_val_indexes, 'val')
			val_loss = InteractTrain(ABatch_val, QBatch_val, AEncoder, ADecoder, QEncoder, QDecoder, ABot_embedding_layer, QBot_embedding_layer, sampler, params, ABot_optimizer, QBot_optimizer, criterion, numSL, batch_mode=False, onlyForward=True)
			val_tokens += val_loss[1]
			avgvalloss += val_loss[0]
		print ("Epoch:",epoch, "\tTime:", time.time() - start, "\tAverage Loss Per Batch::", avgloss/tokens, "\tAverage Validation Loss:", avgvalloss/val_tokens)
		torch.save({'epoch': epoch ,'AEncoder': AEncoder.state_dict(),'ADecoder': ADecoder.state_dict(), 'QEncoder':QEncoder.state_dict(), 'QDecoder':QDecoder.state_dict(), 'ABot_embedding_layer':ABot_embedding_layer.state_dict(), 'QBot_embedding_layer': QBot_embedding_layer.state_dict(), 'ABot_optimizer':ABot_optimizer.state_dict(), 'QBot_optimizer':QBot_optimizer.state_dict()}, "outputs/RL_"+str(epoch))

		loss_arr = np.array(loss_arr)
		np.save(open('outputs/loss_RL_'+str(epoch), 'wb+'), loss_arr)


def InteractTrain(ABatch, QBatch, AEncoder, ADecoder, QEncoder, QDecoder, ABot_embedding_layer, QBot_embedding_layer, sampler, params, ABot_optimizer, QBot_optimizer, criterion, numSL, batch_mode=False, volatile=False, num_dialogs=10, onlyForward=False):
	average_loss = 0.0
	token_count = 0.0
	batch_size = ABatch['questions'].shape[0]
	QBot_input_answer = QBatch['answers'][:,numSL,:]
	QBot_answer_batch_len = QBatch['answers_length'][:,numSL].astype(np.int)
	QBot_history_batch = QBatch['history'][:,:numSL+1,:].astype(np.int)
	QBot_history_batch[:,0,:] = 0
	QBot_history_batch_len = QBatch['history_length'][:,:numSL+1].astype(np.int)
	ABot_history_batch = ABatch['history'][:,:numSL+1,:].astype(np.int)
	ABot_history_batch[:,0,:] = 0
	ABot_history_batch_len = ABatch['history_length'][:,:numSL+1].astype(np.int)
	image_batch = QBatch['images'].astype(np.float)

	image_batch_tensor = Variable(torch.from_numpy(image_batch).float(), volatile=volatile)
	generated_question_length = QBot_input_answer.shape[1]
	generated_answer_length = QBot_history_batch.shape[2] - generated_question_length

	if params['USE_CUDA']:
		image_batch_tensor = image_batch_tensor.cuda(gpu)

	generated_answers = []
	generated_questions = []
	for dialog_num in range(numSL, num_dialogs):
		if dialog_num > numSL:
			QBot_input_answer = ABot_sampled_words.data.cpu().numpy()
			QBot_answer_batch_len = ABot_generated_answer_length_tensor.data.cpu().numpy()
			ABot_input_question = QBot_sampled_words.data.cpu().numpy()
			ABot_question_batch_len = QBot_generated_question_length_tensor.data.cpu().numpy()
			current_fact = np.zeros(shape=(QBot_history_batch.shape[0], 1, QBot_history_batch.shape[2]))
			current_fact_length = np.zeros(shape=(QBot_history_batch.shape[0], 1))
			for example in range(batch_size):
				lenQ = ABot_question_batch_len[example]
				lenA = QBot_answer_batch_len[example]
				current_fact[example, 0, :lenQ] =  ABot_input_question[example, :lenQ]
				current_fact[example, 0, lenQ:lenQ + lenA] = QBot_input_answer[example, :lenA]
				current_fact_length[example,:] = lenQ + lenA
			QBot_history_batch = np.concatenate((QBot_history_batch, current_fact), 1)
			ABot_history_batch = np.concatenate((ABot_history_batch, current_fact), 1)
			QBot_history_batch_len = np.concatenate((QBot_history_batch_len, current_fact_length),1)
			ABot_history_batch_len = np.concatenate((ABot_history_batch_len, current_fact_length),1)

		#QBot Encoder
		QBot_noise_input = Variable(torch.FloatTensor(generated_question_length, batch_size, params['vocab_size']).uniform_(0,1), volatile=volatile)
		ABot_noise_input = Variable(torch.FloatTensor(generated_answer_length, batch_size, params['vocab_size']).uniform_(0,1), volatile=volatile)
		QBot_history_batch_tensor = Variable(torch.from_numpy(QBot_history_batch).long(), volatile=volatile)
		ABot_history_batch_tensor = Variable(torch.from_numpy(ABot_history_batch).long(), volatile=volatile)
		QBot_answer_batch_tensor = Variable(torch.from_numpy(QBot_input_answer).long(), volatile=volatile)
		QBot_answer_batch_length_tensor = Variable(torch.from_numpy(QBot_answer_batch_len).long(), volatile=volatile)
		QBot_history_batch_length_tensor = Variable(torch.from_numpy(QBot_history_batch_len).long(), volatile=volatile)
		ABot_history_batch_length_tensor = Variable(torch.from_numpy(ABot_history_batch_len).long(), volatile=volatile)
		if params['USE_CUDA']:
			QBot_answer_batch_tensor = QBot_answer_batch_tensor.cuda(gpu)
			QBot_history_batch_tensor = QBot_history_batch_tensor.cuda(gpu)
			QBot_noise_input = QBot_noise_input.cuda(gpu)
			ABot_noise_input = ABot_noise_input.cuda(gpu)
			ABot_history_batch_tensor = ABot_history_batch_tensor.cuda(gpu)
			QBot_answer_batch_length_tensor = QBot_answer_batch_length_tensor.cuda(gpu)
			QBot_history_batch_length_tensor = QBot_history_batch_length_tensor.cuda(gpu)
			ABot_history_batch_length_tensor = ABot_history_batch_length_tensor.cuda(gpu)

		QBot_answer_batch_embedding = QBot_embedding_layer(QBot_answer_batch_tensor)
		QBot_fact_batch_embedding = QBot_embedding_layer(QBot_history_batch_tensor.view(-1, QBot_history_batch_tensor.size(2)))
		QBot_history_batch_length_tensor = QBot_history_batch_length_tensor.view(-1,)
		QBot_final_history_encoding, QBot_question_batch_encoder_hidden, QBot_combined_embedding = QEncoder(QBot_answer_batch_embedding, QBot_answer_batch_length_tensor, QBot_fact_batch_embedding, QBot_history_batch_length_tensor, batch_mode=batch_mode)
		_ , QBot_question_decoder_batch_hidden, _ = QDecoder(QBot_final_history_encoding, QBot_question_batch_encoder_hidden, batch_mode=batch_mode)

		#Gumble Sample
		QBot_input_tokens = Variable(torch.ones(batch_size, 1).long()*params['word2ind']['<START>'])
		if params['USE_CUDA']:
			QBot_input_tokens = QBot_input_tokens.cuda(gpu)
		QBot_input_words = QBot_embedding_layer(QBot_input_tokens)
		QBot_sampled_words = []
		QBot_one_hot = []
		next_indices_mask = QBot_input_tokens != params['word2ind']['<END>']
		next_indices_mask = next_indices_mask.long()
		QBot_sample_length = []
		for index in range(generated_question_length):
			probabilities, QBot_question_decoder_batch_hidden, _ = QDecoder(QBot_input_words, QBot_question_decoder_batch_hidden, batch_mode=batch_mode)
			noise = QBot_noise_input.narrow(0,index,1).squeeze(0)
			one_hot, next_indices = sampler(F.log_softmax(probabilities, dim=2).squeeze(1), noise, params['temperature'])
			next_indices_mask = next_indices_mask * (next_indices != params['word2ind']['<END>']).transpose(0,1).long()
			next_indices = next_indices.transpose(0,1) * next_indices_mask
			QBot_sample_length.append(next_indices_mask)
			one_hot = one_hot * next_indices_mask.float().expand_as(one_hot)
			QBot_one_hot.append(one_hot.unsqueeze(1))
			QBot_sampled_words.append(next_indices)
			QBot_input_words = torch.mm(one_hot, QBot_embedding_layer.embedding_layer.weight).unsqueeze(1)
		QBot_sampled_words = torch.cat(QBot_sampled_words, 1)
		QBot_one_hot = torch.cat(QBot_one_hot, 1)

		#ABot Encoder
		QBot_generated_question_length_tensor = torch.sum(torch.cat(QBot_sample_length, 1),1)
		ABot_question_batch_embedding = torch.mm(QBot_one_hot.view(-1, params['vocab_size']),ABot_embedding_layer.embedding_layer.weight).view(QBot_one_hot.size(0), QBot_one_hot.size(1), -1)
		ABot_fact_batch_embedding = ABot_embedding_layer(ABot_history_batch_tensor.view(-1, ABot_history_batch_tensor.size(2)))
		ABot_history_batch_length_tensor = ABot_history_batch_length_tensor.view(-1,)

		ABot_final_history_encoding, ABot_question_batch_encoder_hidden = AEncoder(ABot_question_batch_embedding, QBot_generated_question_length_tensor, ABot_fact_batch_embedding, ABot_history_batch_length_tensor, image_batch_tensor, batch_mode=batch_mode)
		_ , ABot_answer_decoder_batch_hidden = ADecoder(ABot_final_history_encoding, ABot_question_batch_encoder_hidden, batch_mode=batch_mode)
		# answer_input_batch_embedding = ABot_embedding_layer(answer_input_batch_tensor)
		ABot_input_tokens = Variable(torch.ones(batch_size, 1).long()*params['word2ind']['<START>'])
		if params['USE_CUDA']:
			ABot_input_tokens = ABot_input_tokens.cuda(gpu)
		ABot_input_words = ABot_embedding_layer(ABot_input_tokens)
		ABot_sampled_words = []
		ABot_one_hot_samples = []
		ABot_next_indices_mask = ABot_input_tokens != params['word2ind']['<END>']
		ABot_next_indices_mask = ABot_next_indices_mask.long()
		ABot_sample_length = []

		for index in range(generated_answer_length):
			ABot_probabilities, ABot_answer_decoder_batch_hidden = ADecoder(ABot_input_words, ABot_answer_decoder_batch_hidden, batch_mode=batch_mode)
			ABot_noise = ABot_noise_input.narrow(0,index,1).squeeze(0)
			# values, indices = torch.topk(ABot_probabilities, 5, dim=2, largest=True, sorted=False)
			# bias_tensor.zero_()
			# bias_tensor.scatter_(2,indices.transpose(1,2),10)
			# bias_tensor_sum = torch.sum(bias_tensor,1) + 1
			# ABot_probabilities = ABot_probabilities.squeeze(1)
			ABot_one_hot, ABot_next_indices = sampler(F.log_softmax(ABot_probabilities, dim=2).squeeze(1), ABot_noise, params['temperature'])
			ABot_next_indices_mask = ABot_next_indices_mask * (ABot_next_indices != params['word2ind']['<END>']).transpose(0,1).long()
			ABot_next_indices = ABot_next_indices.transpose(0,1) * ABot_next_indices_mask
			ABot_sample_length.append(ABot_next_indices_mask)
			ABot_one_hot = ABot_one_hot * ABot_next_indices_mask.float().expand_as(ABot_one_hot)
			ABot_one_hot_samples.append(ABot_one_hot.unsqueeze(1))
			ABot_sampled_words.append(ABot_next_indices)
			ABot_input_words = torch.mm(ABot_one_hot, ABot_embedding_layer.embedding_layer.weight).unsqueeze(1)

		ABot_sampled_words = torch.cat(ABot_sampled_words, 1)
		# print ([[params['ind2word'][x] for x in ABot_sampled_words[i].data.cpu().numpy() if x != 0] for i in range(batch_size)])
		ABot_one_hot_samples = torch.cat(ABot_one_hot_samples, 1)
		ABot_generated_answer_length_tensor = torch.sum(torch.cat(ABot_sample_length, 1),1)
		QBot_fact_batch_embedding = QBot_fact_batch_embedding.view(batch_size, dialog_num + 1, -1, params['embedding_size'])
		QBot_history_batch_length_tensor = QBot_history_batch_length_tensor.view(batch_size, dialog_num + 1)
		QBot_new_answer_batch_embedding = torch.mm(ABot_one_hot_samples.view(-1, params['vocab_size']),QBot_embedding_layer.embedding_layer.weight).view(ABot_one_hot_samples.size(0), ABot_one_hot_samples.size(1), -1)
		QBot_question_batch_embedding = torch.mm(QBot_one_hot.view(-1, params['vocab_size']),QBot_embedding_layer.embedding_layer.weight).view(QBot_one_hot.size(0), QBot_one_hot.size(1), -1)
		QBot_current_fact_batch_embedding = torch.cat((QBot_question_batch_embedding, QBot_new_answer_batch_embedding), 1)
		QBot_new_fact_batch_embedding = torch.cat((QBot_fact_batch_embedding, QBot_current_fact_batch_embedding.unsqueeze(1)), 1)
		QBot_new_fact_batch_embedding = QBot_new_fact_batch_embedding.view(-1, QBot_new_fact_batch_embedding.size(2), params['embedding_size'])
		QBot_new_fact_length_tensor = ABot_generated_answer_length_tensor + generated_question_length
		QBot_new_history_batch_length_tensor = torch.cat((QBot_history_batch_length_tensor, QBot_new_fact_length_tensor.unsqueeze(1)),1)
		QBot_new_history_batch_length_tensor = QBot_new_history_batch_length_tensor.view(-1,)
		QBot_new_final_history_encoding, QBot_new_question_batch_encoder_hidden, QBot_new_combined_embedding = QEncoder(QBot_new_answer_batch_embedding, ABot_generated_answer_length_tensor, QBot_new_fact_batch_embedding, QBot_new_history_batch_length_tensor, batch_mode=batch_mode)
		image_output = QDecoder.image_decoder(QBot_new_combined_embedding)
		loss_mse = criterion['MSELoss'](image_output.squeeze(0), image_batch_tensor)
		print (loss_mse)
		if not onlyForward:
			loss_mse.backward()
			QBot_optimizer.step()
			ABot_optimizer.step()
		average_loss += loss_mse.data.cpu().numpy()[0]

	return average_loss, (num_dialogs - numSL)*batch_size

train(data, params, AEncoder, ADecoder, QEncoder, QDecoder, ABot_embedding_layer, QBot_embedding_layer, sampler, criterion, ABot_optimizer, QBot_optimizer, current_epoch=0)
