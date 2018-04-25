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

# torch.backends.cudnn.enabled = False
# # random.seed(32)
# # np.random.seed(32)
# # torch.manual_seed(7)
# # torch.cuda.manual_seed_all(7)
# #Load Data
# dialog_loc = '../../../../CS532L-Project/chat_processed_data.h5'
# param_loc = '../../../../CS532L-Project/chat_processed_params.json'
# image_loc = '../../../../CS532L-Project/data_img.h5'

# data = dataloader.DataLoader(dialog_loc, image_loc, param_loc)
# print ("Done: Data Preparation")

#CUDA
USE_CUDA = False
gpu = 0

# #Parameters
# params = {}
# params['batch_first'] = False
# params['num_layers'] = 2
# params['hidden_dim'] = 512
# params['embed_size'] = 300
# params['vocab_size'] = len(data.ind2word.keys())
# params['embedding_size'] = 300
# params['vgg_out'] = 4096
# params['image_embed_size'] = 300
# params['batch_size']=150
# params['epochs'] = 40
# params['rnn_type'] = 'LSTM'
# params['num_dialogs'] = 10
# params['sampleWords'] = False
# params['temperature'] = 0.3
# params['beamSize'] = 5
# params['beamLen'] = 20
# params['word2ind'] = data.word2ind
# params['ind2word'] = data.ind2word
# params['USE_CUDA'] = True
# params['gpu'] = gpu



# compute_ranks = False
# # current_epoch_ABot = 25
# # current_epoch_QBot = 20



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

class AnswerEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, rnn='LSTM', num_layers=1, batch_first=True):
        super(AnswerEncoder, self).__init__()
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



linear_img_layer_dict = torch.load('outputs/linear_300.tar')
# ImgEmbedLayer = LinearL()

# ImgEmbedLayer.load_state_dict(linear_img_layer_dict)
# if USE_CUDA:
#     ImgEmbedLayer = ImgEmbedLayer.cuda(gpu)

class LinearL(nn.Module):
    def __init__(self):
        super(LinearL, self).__init__()
        self.linear = nn.Linear(4096,300)

    def forward(self, X):
        out = F.tanh(self.linear(X))
        return out

class Discriminator_Img(nn.Module):
    def __init__(self, params):
        super(Discriminator_Img, self).__init__()
        self.params = params
#         embedding_weights = np.random.random((params['vocab_size'], params['embed_size']))
#         embedding_weights[0,:] = np.zeros((1, params['embed_size']))
        self.img_embed_layer = LinearL()
        self.img_embed_layer.load_state_dict(linear_img_layer_dict)
        for param in self.img_embed_layer.parameters():
            param.requires_grad = False

        self.linear_ans_vocab_layer = nn.Linear(params['vocab_size'],params['embed_size'])
        self.linear_ques_vocab_layer = nn.Linear(params['vocab_size'],params['embed_size'])
#         self.ans_embedding_layer = ABot.EmbeddingLayer(embedding_weights)
#         self.ques_embedding_layer = ABot.EmbeddingLayer(embedding_weights)
        self.question_encoder = QuestionEncoder(params['embedding_size'], params['hidden_dim'], num_layers=params['num_layers'], rnn=params['rnn_type'], batch_first=params['batch_first'])
        self.answer_encoder = AnswerEncoder(params['embedding_size'], params['hidden_dim'], num_layers=params['num_layers'], rnn=params['rnn_type'], batch_first=params['batch_first'])
        self.linear1 = nn.Linear(2*params['num_layers']*params['hidden_dim'] + params['image_embed_size'], params['linear_hidden'])
        self.activation = getattr(nn, params['activation'])()
        self.linear2 = nn.Linear(params['linear_hidden'], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, question_batch, question_batch_len, answer_batch, answer_batch_len, batch_size, images_tensor):
        batch_mode = True
        sort_index = 0 if self.params['batch_first'] else 1

        question_batch_embedding = self.linear_ques_vocab_layer(question_batch)
        answer_batch_embedding = self.linear_ans_vocab_layer(answer_batch)
#         question_batch_embedding = self.ques_embedding_layer(question_batch)
#         answer_batch_embedding = self.ans_embedding_layer(answer_batch)
        #print("answer_batch_embedding", answer_batch_embedding.size())
        question_batch_encoder_hidden = self.question_encoder.initHidden(batch_size=batch_size)
        answer_batch_encoder_hidden = self.answer_encoder.initHidden(batch_size=batch_size)
        assert(batch_size == answer_batch_embedding.size(0))
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


        #Just taking two hidden state
        #print(question_batch_encoder_hidden[0].size())
        question_batch_encoder_hidden = question_batch_encoder_hidden[0]
        answer_batch_encoder_hidden = answer_batch_encoder_hidden[0]
        assert(self.params['num_layers'] == 2)
        mlp_ques_input = question_batch_encoder_hidden.narrow(0,0,1).view(batch_size,-1)
        mlp_ans_input = answer_batch_encoder_hidden.narrow(0,0,1).view(batch_size,-1)

        #print("mlp_ans", mlp_ans_input.size())
        #print("mlp_ques", mlp_ques_input.size())

        for l in range(1,self.params['num_layers']):
            mlp_ques_input = torch.cat((mlp_ques_input, question_batch_encoder_hidden.narrow(0, l, 1).view(batch_size,-1)),1)
            mlp_ans_input = torch.cat((mlp_ans_input, answer_batch_encoder_hidden.narrow(0,l,1).view(batch_size,-1)),1)

        #print("mlp_ans", mlp_ans_input.size())
        #print("mlp_ques", mlp_ques_input.size())

        # ImagesVar = Variable(torch.from_numpy(batch_i['images']), requires_grad = False).cuda(gpu)
        outImages = self.img_embed_layer(images_tensor)
        #print(out.size())

        mlp_input = torch.cat((mlp_ques_input, mlp_ans_input),1)
        #print("mlp_input", mlp_input.size())
        mlp_input = torch.cat((mlp_input, outImages),1)

        #print("mlp_input", mlp_input.size())


        mlp_out = self.linear1(mlp_input)
        #print("mlp_out", mlp_out.size())
        mlp_out = self.activation(mlp_out)
        mlp_out = self.linear2(mlp_out)
        #print("mlp_out", mlp_out.size())
        mlp_out = self.sigmoid(mlp_out)
        return mlp_out

def getOneHot(output_batch_tensor):
    #print(output_batch_tensor.size())
    output_batch_tensor.unsqueeze_(2)
    y_onehot_2 = torch.FloatTensor(output_batch_tensor.size(0), output_batch_tensor.size(1), params['vocab_size'])
    y_onehot_2.zero_()
    #print("val_b",y_onehot[0,0,3679])
    y_onehot_2.scatter_(2,output_batch_tensor.data,1)

#     for i in range(y_onehot.size(0)):
#         y_onehot[i].scatter_(1, output_batch_tensor.data[i], 1)
    return y_onehot_2

def LoadDiscriminator(params):
    params['activation'] = 'Sigmoid'
    params['linear_hidden'] = 300
    D_img = Discriminator_Img(params)
    d_img_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, D_img.parameters()), lr=0.001, betas=(0.5, 0.999))
    BCE_loss = nn.BCELoss()

    USE_CUDA = True

    checkpoint = torch.load('outputs/discr_0')
    D_img.load_state_dict(checkpoint['discriminator'])
    d_img_optimizer.load_state_dict(checkpoint['optimizer'])
    # for param in D_img.img_embed_layer.parameters():
    #     param.requires_grad = True
    # #d_img_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, D_img.parameters()), lr=0.001, betas=(0.5, 0.999))
    # d_img_optimizer = torch.optim.Adam(D_img.parameters(), lr=0.001, betas=(0.5, 0.999))


    if params['USE_CUDA']:
            D_img = D_img.cuda(gpu)

    print(USE_CUDA)
    if params['USE_CUDA']:
        D_img = D_img.cuda(gpu)
        BCE_loss = BCE_loss.cuda(gpu)
    return D_img, d_img_optimizer, BCE_loss

def train_discriminator(ABatch, QBatch, discr, d_img_optimizer, sampledAnswers,ans_sample_len, sampledQuestions, ques_sample_len,params):

    question_batch = ABatch['questions'][:,dialog_num,:].astype(np.int)
    question_batch_len = ABatch['questions_length'][:,dialog_num].astype(np.int)
    answer_batch_len = QBatch['answers_length'][:,dialog_num].astype(np.int)
    answer_output_batch = QBatch['answers'][:,dialog_num+1,:].astype(np.int)
    question_batch_tensor = Variable(torch.from_numpy(question_batch).long(), volatile=volatile, requires_grad=False)
    question_batch_length_tensor = Variable(torch.from_numpy(question_batch_len).long(), volatile=volatile)
    ques_sample_len = Variable(torch.from_numpy(ques_sample_len).long(), volatile=volatile)
    answer_batch_length_tensor = Variable(torch.from_numpy(answer_batch_len).long(), volatile=volatile)
    ans_sample_len = Variable(torch.from_numpy(ans_sample_len).long(), volatile=volatile)

    answer_output_batch_tensor = Variable(torch.from_numpy(answer_output_batch).long(), volatile=volatile, requires_grad=False)
    zeros_v = Variable(torch.zeros(params['batch_size'],params['beamLen'] - answer_output_batch_tensor.size(1)).long())
    answer_output_batch_tensor = torch.cat((answer_output_batch_tensor, zeros_v),1)
    answer_output_batch_tensor = Variable(getOneHot(answer_output_batch_tensor), volatile = volatile, requires_grad=False)
    question_batch_tensor = Variable(getOneHot(question_batch_tensor), volatile = volatile, requires_grad=False)
    if USE_CUDA:
        question_batch_tensor = question_batch_tensor.cuda(gpu)
        question_batch_length_tensor = question_batch_length_tensor.cuda(gpu)
        answer_batch_length_tensor = answer_batch_length_tensor.cuda(gpu)
        ques_sample_len = ques_sample_len.cuda(gpu)
        answer_output_batch_tensor = answer_output_batch_tensor.cuda(gpu)
        zeros_v = zeros_v.cuda(gpu)
        answer_output_batch_tensor = answer_output_batch_tensor.cuda(gpu)
        ans_sample_len = ans_sample_len.cuda(gpu)
        sampledQuestions = sampledQuestions.cuda(gpu)
        sampledAnswers = sampledAnswers.cuda(gpu)

    volatile = False

    d_img_optimizer.zero_grad()
    batch_size = params['batch_size']

    #print(d_real_result)
    y_real = Variable(torch.ones(batch_size))
    y_fake = Variable(torch.zeros(batch_size))

    if USE_CUDA:
        y_real = y_real.cuda(gpu)
        y_fake = y_fake.cuda(gpu)

    d_real_result = discr(question_batch_tensor, question_batch_length_tensor, answer_output_batch_tensor, answer_batch_length_tensor, params['batch_size'], batch).squeeze()
    d_real_loss = BCE_loss(d_real_result, y_real)

    d_fake_result = discr(sampledQuestions, ques_sample_len, sampledAnswers, ans_sample_len, params['batch_size'], batch).squeeze()
    d_fake_loss = BCE_loss(d_fake_result, y_fake)

    d_train_loss =  d_real_loss + d_fake_loss


    d_train_loss.backward()
    d_img_optimizer.step()
    #d_losses.append(d_train_loss.data[0])

    print("D: loss: R: ",d_real_loss.data[0],"F: ",d_fake_loss.data[0], "C: ", d_train_loss.data[0])

    # g_fake_result = discr(sampledQuestions, ques_sample_len, sampledAnswers, ans_sample_len, params['batch_size'], batch).squeeze()

#     g_fake_loss = BCE_loss(g_fake_result, y_real)

#     g_train_loss =  g_fake_loss

#     print("G: loss", g_train_loss.data[0])
    return

def get_discriminator_loss(discr, sampledAnswers, ans_sample_len, sampledQuestions, ques_sample_len, images_tensor, params, BCE_loss, batch_size=None):
    volatile = False
    if batch_size is None:
        batch_size = params['batch_size']
    y_real = Variable(torch.ones(batch_size))
    y_fake = Variable(torch.zeros(batch_size))

    if params['USE_CUDA']:
        y_real = y_real.cuda(gpu)
        y_fake = y_fake.cuda(gpu)
#     if not gen_opt:

#         d_real_result = discr(question_batch_tensor, question_batch_length_tensor, answer_output_batch_tensor, answer_batch_length_tensor, params['batch_size'], batch).squeeze()
#         d_real_loss = BCE_loss(d_real_result, y_real)

#         d_fake_result = discr(sampledQuestions, ques_sample_len, sampledAnswers, ans_sample_len, params['batch_size'], batch).squeeze()
#         d_fake_loss = BCE_loss(d_fake_result, y_fake)

#         d_train_loss =  d_real_loss + d_fake_loss


#         d_train_loss.backward()
#         d_img_optimizer.step()
#         d_losses.append(d_train_loss.data[0])

#         print("D: loss: R: ",d_real_loss.data[0],"F: ",d_fake_loss.data[0], "C: ", d_train_loss.data[0])

    g_fake_result = discr(sampledQuestions, ques_sample_len, sampledAnswers, ans_sample_len, batch_size, images_tensor).squeeze()
    g_fake_loss = BCE_loss(g_fake_result, y_real)
    g_train_loss =  g_fake_loss
    return g_train_loss
