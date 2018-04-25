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
import discriminator as Discriminator

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
current_epoch_ABot = 26
current_epoch_QBot = 23

#Define Models
AEncoder = ABot_Encoder.ABotEncoder(params)
ADecoder = ABot_Decoder.ABotDecoder(params)
QEncoder = QBot_Encoder.QBotEncoder(params)
QDecoder = QBot_Decoder.QBotDecoder(params)
embedding_weights = np.random.random((params['vocab_size'], params['embed_size']))
embedding_weights[0,:] = np.zeros((1, params['embed_size']))
ABot_embedding_layer = ABot.EmbeddingLayer(embedding_weights)
QBot_embedding_layer = QBot.EmbeddingLayer(embedding_weights)
sampler = ABot.GumbelSampler()
embedding_weights_discr = np.random.random((params['vocab_size'], params['embed_size']))
embedding_weights_discr[0,:] = np.zeros((1, params['embed_size']))
print (embedding_weights_discr)
discriminator = Discriminator.Discriminator(params, embedding_weights_discr)

#Criterion
criterion = {}
criterion['CrossEntropyLoss'] = nn.CrossEntropyLoss(reduce=False)
criterion['HingeEmbeddingLoss'] = nn.HingeEmbeddingLoss(margin=0.0, size_average=False)
criterion['MSELoss'] = nn.MSELoss(size_average=False)
criterion['BCELoss'] = nn.BCELoss(size_average=False)

#Optimizer
ABot_optimizer = torch.optim.Adam([{'params':AEncoder.parameters()},{'params':ADecoder.parameters()},{'params':ABot_embedding_layer.parameters()}], lr=0.001)
QBot_optimizer = torch.optim.Adam([{'params':QEncoder.parameters()},{'params':QDecoder.parameters()},{'params':QBot_embedding_layer.parameters()}], lr=0.001)
discriminator_optimizer = torch.optim.Adam([{'params':discriminator.parameters()}], lr=0.001)
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
    discriminator = discriminator.cuda(gpu)
    for key in crtierion.keys():
        criterion[key] = criterion[key].cuda(gpu)

def TrainDiscriminator(data, params, AEncoder, ADecoder, QEncoder, QDecoder, ABot_embedding_layer, QBot_embedding_layer, sampler, discriminator, criterion, discriminator_optimizer, current_epoch=0):
    number_of_batches = math.ceil(data.datasize['train']/params['batch_size'])
    number_of_val_batches = math.ceil(data.datasize['val']/params['batch_size'])
    indexes = np.arange(data.datasize['train'])
    val_indexes = np.arange(data.datasize['val'])

    print (number_of_batches, number_of_val_batches)
    discriminator.train()
    np.random.shuffle(indexes)
    start = time.time()
    avg_loss_arr = []
    loss_arr = []
    avgvalloss = 0.0
    val_tokens = 0.0
    avgloss = 0.0
    tokens = 0.0

    for epoch in range(params['epochs']):
        for batch_num in range(number_of_batches):
            batch_indexes = indexes[batch_num*params['batch_size']:(batch_num+1)*params['batch_size']]
            ABatch = data.getBatch(batch_indexes, 'train')
            QBatch = data.getQBatch(batch_indexes, 'train')
            true_questions = ABatch['questions'][:,:,:]
            true_answers = QBatch['answers'][:,1:,:]
            
            print (true_questions)
            print (true_answers)
            aa

TrainDiscriminator(data, params, AEncoder, ADecoder, QEncoder, QDecoder, ABot_embedding_layer, QBot_embedding_layer, sampler, discriminator, criterion, discriminator_optimizer)
