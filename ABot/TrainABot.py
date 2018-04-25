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
current_epoch = 26

#Define Models
AEncoder = ABot_Encoder.ABotEncoder(params)
ADecoder = ABot_Decoder.ABotDecoder(params)
embedding_weights = np.random.random((params['vocab_size'], params['embed_size']))
embedding_weights[0,:] = np.zeros((1, params['embed_size']))
print (embedding_weights)
embedding_layer = ABot.EmbeddingLayer(embedding_weights)
sampler = ABot.GumbelSampler()


#Criterion
criterion = {}
criterion['CrossEntropyLoss'] = nn.CrossEntropyLoss(reduce=False)
criterion['HingeEmbeddingLoss'] = nn.HingeEmbeddingLoss(margin=0.0, size_average=False)

#Optimizer
optimizer = torch.optim.Adam([{'params':AEncoder.parameters()},{'params':ADecoder.parameters()},{'params':embedding_layer.parameters()}], lr=0.001)

#Load Models
if current_epoch > 0:
	checkpoint = torch.load('outputs/supervised_ABot_HistoryAttention_AttentionRankingLoss_2Layers_'+str(current_epoch-1))
	AEncoder.load_state_dict(checkpoint['AEncoder'])
	ADecoder.load_state_dict(checkpoint['ADecoder'])
	embedding_layer.load_state_dict(checkpoint['embedding_layer'])
	optimizer.load_state_dict(checkpoint['optimizer'])

if USE_CUDA:
	AEncoder = AEncoder.cuda(gpu)
	ADecoder = ADecoder.cuda(gpu)
	embedding_layer = embedding_layer.cuda(gpu)
	sampler = sampler.cuda(gpu)

def supervisedTrain(batch, AEncoder, ADecoder, embedding_layer, sampler, params, optimizer, criterion, onlyForward=False, num_dialogs=10, batch_mode=True, sample=False):
	average_loss = 0.0
	token_count = 0.0
	batch_size = batch['questions'].shape[0]
	volatile = False
	if onlyForward:
		volatile = True
	for dialog_num in range(num_dialogs):
        #Get Data
		question_batch = batch['questions'][:,dialog_num,:].astype(np.int)
		answer_input_batch = batch['answers_input'][:,dialog_num,:].astype(np.int)
		answer_output_batch = batch['answers_output'][:,dialog_num,:].astype(np.int)
		question_batch_len = batch['questions_length'][:,dialog_num].astype(np.int)
		answer_batch_len = batch['answers_length'][:,dialog_num].astype(np.int)
		history_batch = batch['history'][:,:dialog_num+1,:].astype(np.int)
		history_batch_len = batch['history_length'][:,:dialog_num+1].astype(np.int)
		image_batch = batch['images'].astype(np.float)

		#Create Tensors
		question_batch_tensor = Variable(torch.from_numpy(question_batch).long(), volatile=volatile)
		answer_input_batch_tensor = Variable(torch.from_numpy(answer_input_batch).long(), volatile=volatile)
		answer_output_batch_tensor = Variable(torch.from_numpy(answer_output_batch).long(), volatile=volatile)
		history_batch_tensor = Variable(torch.from_numpy(history_batch).long(), volatile=volatile)
		image_batch_tensor = Variable(torch.from_numpy(image_batch).float(), volatile=volatile)
		question_batch_length_tensor = Variable(torch.from_numpy(question_batch_len).long(), volatile=volatile)
		history_batch_length_tensor = Variable(torch.from_numpy(history_batch_len).long(), volatile=volatile)
		answer_batch_length_tensor = Variable(torch.from_numpy(answer_batch_len).long(), volatile=volatile)
		noise_input = Variable(torch.FloatTensor(params['beamLen'], batch_size, params['vocab_size']).uniform_(0,1), volatile=volatile)

		if USE_CUDA:
			question_batch_tensor = question_batch_tensor.cuda(gpu)
			answer_input_batch_tensor = answer_input_batch_tensor.cuda(gpu)
			answer_output_batch_tensor = answer_output_batch_tensor.cuda(gpu)
			history_batch_tensor = history_batch_tensor.cuda(gpu)
			image_batch_tensor = image_batch_tensor.cuda(gpu)
			question_batch_length_tensor = question_batch_length_tensor.cuda(gpu)
			history_batch_length_tensor = history_batch_length_tensor.cuda(gpu)
			answer_batch_length_tensor = answer_batch_length_tensor.cuda(gpu)
			noise_input = noise_input.cuda(gpu)

		question_batch_embedding = embedding_layer(question_batch_tensor)
		fact_batch_embedding = embedding_layer(history_batch_tensor.view(-1, history_batch_tensor.size(2)))
		history_batch_length_tensor = history_batch_length_tensor.view(-1,)
		# fact_batch_embedding = fact_batch_embedding.view(-1, dialog_num+1, fact_batch_embedding.size(1), fact_batch_embedding.size(2))

		final_history_encoding, question_batch_encoder_hidden = AEncoder(question_batch_embedding, question_batch_length_tensor, fact_batch_embedding, history_batch_length_tensor, image_batch_tensor, batch_mode=batch_mode)
		_ , answer_decoder_batch_hidden = ADecoder(final_history_encoding, question_batch_encoder_hidden, batch_mode=batch_mode)
		answer_input_batch_embedding = embedding_layer(answer_input_batch_tensor)
		probabilities_answer_batch, _= ADecoder(answer_input_batch_embedding, answer_decoder_batch_hidden, batch_mode=batch_mode)
		#
		# input_words = embedding_layer(answer_input_batch_tensor.narrow(1,0,1))
		# generated_answer_decoder_batch_hidden = answer_decoder_batch_hidden
		# for index in range(params['beamLen']):
		# 	probabilities, generated_answer_decoder_batch_hidden, generated_answer_embedding = ADecoder(input_words, generated_answer_decoder_batch_hidden, batch_mode=batch_mode)
		# 	noise = noise_input.narrow(0,index,1).squeeze(0)
		# 	one_hot, next_indices = sampler(F.log_softmax(probabilities, dim=2).squeeze(1), noise, params['temperature'])
		# 	next_indices_mask = next_indices != params['word2ind']['<END>']
		# 	next_indices = next_indices * next_indices_mask.long()
		# 	input_words = embedding_layer(next_indices)
		# 	input_words = input_words.transpose(0,1)

		#Cross Entropy Loss
		probabilities_answer_batch = probabilities_answer_batch.view(-1, probabilities_answer_batch.size(2))
		answer_output_batch_tensor = answer_output_batch_tensor.view(-1,)
		loss = criterion['CrossEntropyLoss'](probabilities_answer_batch, answer_output_batch_tensor)
		mask = answer_output_batch_tensor != 0
		mask = mask.float()
		loss_masked = torch.mul(loss, mask)
		loss_masked = loss_masked.sum()

		true_answer_decoder_batch_hidden = ADecoder.answer_decoder.initHidden(batch_size=batch_size)
		# answer_batch_length_tensor_sorted, answer_perm_idx = answer_batch_length_tensor.sort(0, descending=True)
		# _, answer_perm_idx_resort = answer_perm_idx.sort(0, descending=False)
		# answer_input_batch_embedding = answer_input_batch_embedding.index_select(0,answer_perm_idx)
		if not params['batch_first']:
			answer_input_batch_embedding = answer_input_batch_embedding.transpose(0,1)
		# packed_answer_batch_embedding = nn.utils.rnn.pack_padded_sequence(answer_input_batch_embedding, answer_batch_length_tensor_sorted.data.cpu().numpy(), batch_first=params['batch_first'])
		true_answer_output, true_answer_decoder_batch_hidden = ADecoder.answer_decoder(answer_input_batch_embedding, true_answer_decoder_batch_hidden, isPacked=True)
		if not params['batch_first']:
			true_answer_output = true_answer_output.transpose(0,1)
		# true_answer_output = true_answer_output.index_select(0, answer_perm_idx_resort)
		true_answer_embedding = ADecoder.attention(true_answer_output, answer_input_batch_tensor)

		#Ranking Loss
		final_history_embedding = final_history_encoding.view(-1,params['embed_size'])
		true_answer_embedding = true_answer_embedding.view(-1,params['embed_size'])
		norm1 = torch.norm(true_answer_embedding, 2, 1).unsqueeze(1).expand_as(true_answer_embedding)
		norm2 = torch.norm(final_history_embedding, 2, 1).unsqueeze(1).expand_as(final_history_embedding)
		dot_product = torch.mm(final_history_embedding/norm2, (true_answer_embedding/norm1).transpose(0,1))
		diagonal = torch.diag(dot_product).unsqueeze(1).expand_as(dot_product)
		difference = diagonal - dot_product
		labels = Variable(torch.eye(difference.size(0)).float()*2 - 1)
		if params['USE_CUDA']:
			labels = labels.cuda(params['gpu'])
		labels = labels.view(-1,)
		difference = difference.view(-1,)
		loss_hinge = criterion['HingeEmbeddingLoss'](difference, labels)

		total_loss = loss_hinge + loss_masked
		if not onlyForward:
			optimizer.zero_grad()
			total_loss.backward()
			optimizer.step()
		#Generation Loss
		# generated_answer_embedding = generated_answer_embedding.view(-1,params['embed_size'])
		# true_answer_embedding = true_answer_embedding.view(-1,params['embed_size'])
		# norm1_g = torch.norm(true_answer_embedding, 2, 1).unsqueeze(1).expand_as(true_answer_embedding)
		# norm2_g = torch.norm(generated_answer_embedding, 2, 1).unsqueeze(1).expand_as(generated_answer_embedding)
		# dot_product_g = torch.mm(true_answer_embedding/norm1_g, (generated_answer_embedding/norm2_g).transpose(0,1))
		# diagonal_g = torch.diag(dot_product_g).unsqueeze(1).expand_as(dot_product_g)
		# difference_g = diagonal_g - dot_product_g
		# labels_g = Variable(torch.eye(difference_g.size(0)).float()*2 - 1)
		# if params['USE_CUDA']:
		# 	labels_g = labels_g.cuda(params['gpu'])
		# labels_g = labels_g.view(-1,)
		# difference_g = difference_g.view(-1,)
		# loss_generation = criterion['HingeEmbeddingLoss'](difference_g, labels_g)

			# parameters = list(AEncoder.parameters())+list(ADecoder.parameters())+list(embedding_layer.parameters())
			# parameters = list(filter(lambda p: p.grad is not None, parameters))
			# for p in parameters:
			# 	p.grad.data.clamp_(-5.0, 5.0)


		average_loss += loss_masked.data.cpu().numpy()[0]
		token_count += mask.sum().data.cpu().numpy()[0]
		#loss_masked = loss_masked/ mask.sum()
        # print (dialog_num)
	return average_loss, token_count

def train(data, params, AEncoder, ADecoder, embedding_layer, sampler, criterion, optimizer, current_epoch=0):
    number_of_batches = math.ceil(data.datasize['train']/params['batch_size'])
    number_of_val_batches = math.ceil(data.datasize['val']/params['batch_size'])
    # indexes = np.arange(data.datasize['train'])
    indexes = np.arange(data.datasize['train'])
    val_indexes = np.arange(data.datasize['val'])

    print (number_of_batches, number_of_val_batches)
    np.random.seed(1234)
    lr = 0.002
    lr_decay = 0.9997592083
    #lr_decay = 1.0
    min_lr = 5e-5

    AEncoder.train()
    ADecoder.train()
    embedding_layer.train()
    print("Current Epoch:",current_epoch)
    for epoch in range(current_epoch, params['epochs']):
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
            #batch_indexes = np.random.randint(data.datasize['train'], size=params['batch_size'])
            #if batch_num == 0:
            #batch_indexes = np.array([ 49770,33486,40377,11948,9577,20869,16765,6823,30332,27758,1753,33905,41981,29280,22954,36208,40717,4615,39403,40655]) - 1
            #else:
            #	batch_indexes = np.array([ 29173,41365,36417,36532,17365,20208,44022,21525,12940,25041,32303,34575,25448,166, 35768,41220,15860,26001,38355, 49128]) - 1
            batch = data.getBatch(batch_indexes, 'train')
            loss = supervisedTrain(batch, AEncoder, ADecoder, embedding_layer, sampler, params, optimizer, criterion, batch_mode=True, sample=False)
            #avgloss += loss[0][0]
            tokens += loss[1]
            # if avgloss > 0:
            #     avgloss = 0.95 * avgloss + 0.05 * (loss[0]/tokens)
            # else:
            avgloss += loss[0]
            loss_arr.append(loss[0]/loss[1])
			#if lr > min_lr:
				#lr *= lr_decay
				#for param_group in optimizer.param_groups:
				#	param_group['lr'] = lr
            if batch_num%20 == 0:
				#print ("Done Batch:", batch_num, "\tAverage Loss Per Batch:", avgloss/(batch_num+1), "\t Current Batch Loss: ", loss, "\tlr:",lr)
                print ("Done Batch:", batch_num, "\tTime:",time.time() - start,"\tAverage Loss Per Batch:", avgloss/tokens, "\t Current Batch Loss: ", loss[0]/loss[1], "\tlr:",lr)

        for batch_val_num in range(number_of_val_batches):
            batch_val_indexes = val_indexes[batch_val_num*params['batch_size']:(batch_val_num+1)*params['batch_size']]
            batch_val = data.getBatch(batch_val_indexes, 'val')
            val_loss = supervisedTrain(batch, AEncoder, ADecoder, embedding_layer, sampler, params, optimizer, criterion, onlyForward=True)
            val_tokens += val_loss[1]
            avgvalloss += val_loss[0]
        print ("Epoch:",epoch, "\tTime:", time.time() - start, "\tAverage Loss Per Batch::", avgloss/tokens, "\tAverage Validation Loss:", avgvalloss/val_tokens)
        #torch.save({'epoch': epoch ,'image_encoder': image_encoder.state_dict(),'embedding_layer': embedding_layer.state_dict(), 'question_encoder':question_encoder.state_dict(), 'fact_encoder':fact_encoder.state_dict(), 'history_encoder':history_encoder.state_dict(), 'answer_decoder':answer_decoder.state_dict(), 'optimizer':optimizer.state_dict()}, "outputs/supervised_ABot_"+str(epoch))
        torch.save({'epoch': epoch ,'AEncoder': AEncoder.state_dict(),'ADecoder': ADecoder.state_dict(), 'embedding_layer':embedding_layer.state_dict(), 'optimizer':optimizer.state_dict()}, "outputs/supervised_ABot_HistoryAttention_AttentionRankingLoss_2Layers_"+str(epoch))

        loss_arr = np.array(loss_arr)
        np.save(open('outputs/loss_ABot_HistoryAttention_AttentionRankingLoss_2Layers_'+str(epoch), 'wb+'), loss_arr)


train(data, params, AEncoder, ADecoder, embedding_layer, sampler, criterion, optimizer, current_epoch=current_epoch)
