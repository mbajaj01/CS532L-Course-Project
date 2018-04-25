import json
import numpy as np
import random
import torch
import torch.nn.functional as F
import ABot
import json
import h5py


class DataLoader:
	def __init__(self, dialog_loc, image_loc, param_loc):
		self.dialog_loc = dialog_loc
		self.image_loc = image_loc
		self.param_loc = param_loc
		self.dialog = h5py.File(dialog_loc, 'r')
		self.imageAll = h5py.File(image_loc, 'r')
		with open(param_loc, 'rb') as input_json:
			self.params = json.loads(input_json.read().decode('utf-8'))
		self.processData()

	def getCNNOutputs(self):
		all_images = {'train' : {}, 'val' : {}}
		cnn_features = []
		with open('../visdial_params.json', 'rb') as input_json:
			self.cnnParams = json.loads(input_json.read().decode('utf-8'))
		for dataset in ['train','val']:
			for index, img in enumerate(self.cnnParams['img_'+dataset]):
				all_images[dataset][int(img['imgId'])] = index
		self.cnnFeatures = h5py.File('../vdl_img_vgg.h5', 'r')
		output = h5py.File('../vgg_cnn_features.h5', 'w')
		for dataset in ['test', 'val', 'train']:
			features = np.zeros((len(self.params['unique_img_'+dataset]), self.cnnFeatures['images_train'].shape[1], self.cnnFeatures['images_train'].shape[2], self.cnnFeatures['images_train'].shape[3]))
			print (features.shape)
			if dataset == 'test':
				print (self.cnnFeatures['images_val'].shape)
			else:
				print (self.cnnFeatures['images_train'].shape)
			print ()
			for index, img_id in enumerate(self.params['unique_img_'+dataset]):
				if dataset == 'test':
					features[index] = np.array(self.cnnFeatures['images_val'][all_images['val'][int(img_id)]])
				else:
					features[index] = np.array(self.cnnFeatures['images_train'][all_images['train'][int(img_id)]])
			output.create_dataset('img_'+dataset, data=features)
		output.close()


	def processParams(self):
		self.unique_img_train = self.params['unique_img_train']
		self.unique_img_test = self.params['unique_img_test']
		self.unique_img_val = self.params['unique_img_val']
		self.ind2word = self.params['ind2word']
		self.ind2word = {int(key): value for key,value in self.ind2word.items()}
		self.word2ind = self.params['word2ind']

		#Add tokens
		self.START_TOKEN = '<START>'
		self.END_TOKEN = '<END>'
		self.PAD_TOKEN = '<PAD>'
		self.START_INDEX = len(self.ind2word.keys()) + 1
		self.END_INDEX = len(self.ind2word.keys()) + 2
		self.PAD_INDEX = 0

		self.ind2word[self.PAD_INDEX] = self.PAD_TOKEN
		self.ind2word[self.START_INDEX] = self.START_TOKEN
		self.ind2word[self.END_INDEX] = self.END_TOKEN

		self.word2ind[self.PAD_TOKEN] = self.PAD_INDEX
		self.word2ind[self.START_TOKEN] = self.START_INDEX
		self.word2ind[self.END_TOKEN] = self.END_INDEX

	def processQA(self):
		self.questions = {}
		self.questions_length = {}
		self.questions_count = {}

		self.answers = {}
		self.answers_length = {}
		self.answers_indexes = {}

		self.captions = {}
		self.captions_length = {}

		self.options = {}
		self.options_length = {}
		self.options_list = {}
		self.options_probs = {}

		for dataset in ['train','val','test']:
			self.questions[dataset] = np.array(self.dialog['ques_'+dataset])
			self.questions_length[dataset] = np.array(self.dialog['ques_length_'+dataset])
			self.questions_count[dataset] = np.array(self.dialog['ques_count_'+dataset])

		for dataset in ['train','val','test']:
			self.answers[dataset] = np.array(self.dialog['ans_'+dataset])
			self.answers_length[dataset] = np.array(self.dialog['ans_length_'+dataset])
			self.answers_indexes[dataset] = np.array(self.dialog['ans_index_'+dataset]) - 1

		for dataset in ['train','val','test']:
			self.captions[dataset] = np.array(self.dialog['cap_'+dataset])
			self.captions_length[dataset] = np.array(self.dialog['cap_length_'+dataset])

		for dataset in ['val','test']:
			self.options[dataset] = np.array(self.dialog['opt_'+dataset]) - 1
			self.options_length[dataset] = np.array(self.dialog['opt_length_'+dataset])
			self.options_list[dataset] = np.array(self.dialog['opt_list_'+dataset])
			# self.options_probs[dataset] = np.array(self.dialog['opt_len_'+dataset])

	def processImage(self, normalize):
		self.images = {}
		self.image_pos = {}
		for dataset in ['train','val','test']:
			if normalize:
				self.images[dataset] = np.array(self.imageAll['images_'+dataset]) / (np.sqrt(np.sum(np.square(self.imageAll['images_'+dataset]), 1)))[:,None]
			else:
				self.images[dataset] = np.array(self.imageAll['images_'+dataset])

		for dataset in ['train','val','test']:
			self.image_pos[dataset] = np.array(self.dialog['img_pos_'+dataset])

	def processHistory(self, maxHistoryLen=60):
		self.history = {}
		self.history_length = {}

		for dataset in ['train', 'val', 'test']:
			self.history[dataset] = np.zeros((self.datasize[dataset], self.dialogLength, self.questionLength+self.answerLength), dtype=np.int64)
			self.history_length[dataset] = np.zeros((self.datasize[dataset], self.dialogLength), dtype=np.int64)

			for example in range(self.datasize[dataset]):
				#First round has caption as history
				captionLength = min(self.captions_length[dataset][example], self.questionLength+self.answerLength)
				self.history[dataset][example, 0, :captionLength] = self.captions[dataset][example, :captionLength]
				self.history_length[dataset][example, 0] = captionLength

				#Other Rounds have previous Q + A
				for turn in range(self.dialogLength-1):
					lenQ = self.questions_length[dataset][example, turn]
					lenA = self.answers_length[dataset][example, turn]
					self.history[dataset][example, turn + 1, :lenQ] = self.questions[dataset][example, turn, :lenQ]
					self.history[dataset][example, turn + 1, lenQ:lenQ+lenA] = self.answers[dataset][example, turn, :lenA]
					self.history_length[dataset][example, turn + 1] = lenQ + lenA


	def processAnswers(self):
		self.answers_input = {}
		self.answers_output = {}
		self.appended_answers_length = {}
		for dataset in ['train','val','test']:
			self.answers_input[dataset] = np.zeros((self.datasize[dataset], self.dialogLength, self.answerLength + 1), dtype=np.int64)
			self.answers_output[dataset] = np.zeros((self.datasize[dataset], self.dialogLength, self.answerLength + 1), dtype=np.int64)

			for example in range(self.datasize[dataset]):
				for turn in range(self.dialogLength):
					answerLength = self.answers_length[dataset][example, turn]
					self.answers_input[dataset][example,turn,0] = self.START_INDEX
					self.answers_input[dataset][example,turn,1:answerLength+1] = self.answers[dataset][example,turn,:answerLength]

					self.answers_output[dataset][example, turn, :answerLength] = self.answers[dataset][example,turn,:answerLength]
					self.answers_output[dataset][example, turn, answerLength] = self.END_INDEX

			self.appended_answers_length[dataset] = self.answers_length[dataset] + 1

	def processOptions(self):
		self.options_input = {}
		self.options_output = {}
		self.appended_options_length = {}
		for dataset in ['val','test']:
			self.options_input[dataset] = np.zeros((self.options_list[dataset].shape[0], self.answers[dataset].shape[2] + 1), dtype=np.int64)
			self.options_output[dataset] = np.zeros((self.options_list[dataset].shape[0], self.answers[dataset].shape[2] + 1), dtype=np.int64)

			for example in range(self.options_list[dataset].shape[0]):
				optionLength = self.options_length[dataset][example]
				self.options_input[dataset][example, 0] = self.START_INDEX
				self.options_input[dataset][example, 1:optionLength+1] = self.options_list[dataset][example, :optionLength]

				self.options_output[dataset][example, :optionLength] = self.options_list[dataset][example, :optionLength]
				self.options_output[dataset][example, optionLength] = self.END_INDEX

			self.appended_options_length[dataset] = self.options_length[dataset] + 1


	def processData(self, normalize=True):
		self.isRightAligned = False
		self.processParams()
		self.processQA()
		self.processImage(normalize)
		self.datasize = {}
		for dataset in ['train','val','test']:
			self.datasize[dataset] = self.questions[dataset].shape[0]
		self.dialogLength = self.questions['train'].shape[1]
		self.questionLength = self.questions['train'].shape[2]
		self.answerLength = self.answers['train'].shape[2]
		self.processAnswers()
		self.processHistory()
		self.processOptions()
		# self.rightAlignAll()

	def getBatch(self, indexes, dataset):
		batch = {}
		batch['questions_length'] = self.questions_length[dataset][indexes,:]
		batch_max_ques_length = np.max(batch['questions_length'])
		if self.isRightAligned:
			batch['questions'] = self.questions[dataset][indexes,:,self.questions[dataset].shape[2] - batch_max_ques_length:]
		else:
			batch['questions'] = self.questions[dataset][indexes,:,:batch_max_ques_length]

		batch['history_length'] = self.history_length[dataset][indexes,:]
		batch_max_history_length = np.max(batch['history_length'])
		if self.isRightAligned:
			batch['history'] = self.history[dataset][indexes,:,self.history[dataset].shape[2] - batch_max_history_length:]
		else:
			batch['history'] = self.history[dataset][indexes,:,:batch_max_history_length]

		batch['images'] = self.images[dataset][self.image_pos[dataset][indexes],:]

		batch['answers_length'] = self.appended_answers_length[dataset][indexes,:]
		batch_max_answer_length = np.max(batch['answers_length'])
		batch['answers_input'] = self.answers_input[dataset][indexes,:,:batch_max_answer_length]
		batch['answers_output'] = self.answers_output[dataset][indexes,:,:batch_max_answer_length]
		batch['answers_indexes'] = self.answers_indexes[dataset][indexes,:]

		if dataset == 'test' or dataset == 'val':
			batch['options_indexes_array'] = self.options[dataset][indexes,:,:]
			batch['options_indexes'] = np.reshape(batch['options_indexes_array'], (-1,))
			batch['options_length'] = self.appended_options_length[dataset][batch['options_indexes']]
			batch_max_option_length = np.max(batch['options_length'])
			batch['options_input'] = self.options_input[dataset][batch['options_indexes'],:batch_max_option_length]
			batch['options_output'] = self.options_output[dataset][batch['options_indexes'],:batch_max_option_length]
			batch['options_length'] = np.reshape(batch['options_length'], (batch['options_indexes_array'].shape[0], batch['options_indexes_array'].shape[1], batch['options_indexes_array'].shape[2]))
			batch['options_input'] = np.reshape(batch['options_input'], (batch['options_indexes_array'].shape[0], batch['options_indexes_array'].shape[1], batch['options_indexes_array'].shape[2], -1))
			batch['options_output'] = np.reshape(batch['options_output'], (batch['options_indexes_array'].shape[0], batch['options_indexes_array'].shape[1], batch['options_indexes_array'].shape[2], -1))
		return batch


	def rightAlignAll(self):
		self.isRightAligned = True
		for dataset in ['train', 'test', 'val']:
			self.questions[dataset] = self.rightAlign(self.questions[dataset], self.questions_length[dataset])
			self.history[dataset] = self.rightAlign(self.history[dataset], self.history_length[dataset])

	def rightAlign(self, sequences, length):
		rightAlign = np.zeros_like(sequences)
		dims = len(sequences.shape)
		if dims == 3:
			for example in range(sequences.shape[0]):
				for turn in range(sequences.shape[1]):
					rightAlign[example,turn, sequences.shape[2] - length[example, turn]:] = sequences[example, turn, :length[example, turn]]
		return rightAlign

# dialog_loc = '../chat_processed_data.h5'
# param_loc = '../chat_processed_params.json'
# image_loc = '../data_img.h5'

# a = DataLoader(dialog_loc, image_loc, param_loc)
# a.getTrainBatch(np.random.randint(100, size=30))
