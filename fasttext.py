import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
import argparse
import transformers # pytorch transformers
from transformers import TrainingArguments
from transformers import Trainer
from torch.utils.data.dataloader import default_collate

import math
from datasets import Dataset
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from tqdm import tqdm
import re
import io
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import csv
from collections import defaultdict

from transformers import AutoTokenizer, AutoModelForMaskedLM


import string
punc = string.punctuation + '’—”“‘’"'

def clean(text):
	"""
	- Remove entity mentions (eg. '@united')
	- Correct errors (eg. '&amp;' to '&')
	@param    text (str): a string to be processed.
	@return   text (Str): the processed string.
	"""
	# Remove '@name'
	text = re.sub(r'(@.*?)[\s]', ' ', text)

	# Replace '&amp;' with '&'
	text = re.sub(r'&amp;', '&', text)

	# Remove trailing whitespace
	text = re.sub(r'\s+', ' ', text).strip()

	text = ' '.join([w.lower().strip(' ') for w in text.split(' ')])
	text = ''.join(w for w in text if w not in punc)
	return text.strip(' ')

def eec_builder(eec_dict, tag):

	with open('eec/{}.csv'.format(tag), encoding='utf-8') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		headers = next(csv_reader)
		for row in csv_reader:
			sentence = row[1]
			gender = row[4]
			race = row[5]
			emotion = row[6]
			if emotion == '':
				emotion = 'valence'
			if race == '':
				eec_dict[tag][emotion][gender].append(sentence)
			else:
				eec_dict[tag][emotion][gender + ' ' + race].append(sentence)
				eec_dict[tag][emotion][gender].append(sentence)
				eec_dict[tag][emotion][race].append(sentence)
	return eec_dict

def get_eecs():
	#build all of the eecs
	# key 1: en1, en2, es, ar
	# key 2: tasks
	# key 3: class
	eec_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
	
	eec_dict = eec_builder(eec_dict, 'en')
	eec_dict = eec_builder(eec_dict, 'en_es')
	eec_dict = eec_builder(eec_dict, 'en_ar')
	eec_dict = eec_builder(eec_dict, 'es')
	eec_dict = eec_builder(eec_dict, 'ar')
	
	return eec_dict

def get_eec_dict(language, emotion):
	eec_dict = defaultdict(list)
	with open('eec/{}.csv'.format(language), encoding='utf-8') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		headers = next(csv_reader)
		for row in csv_reader:
			sentence = row[1]
			gender = row[4]
			race = row[5]
			n_emotion = row[6]
			if n_emotion == '':
				n_emotion = 'valence'
			if n_emotion == emotion:
				if race == '':
					eec_dict[gender].append(sentence)
				else:
					eec_dict[gender + ' ' + race].append(sentence)
					eec_dict[gender].append(sentence)
					eec_dict[race].append(sentence)
	return eec_dict

def load_vectors(fname, vocab):
	fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
	n, d = map(int, fin.readline().split())
	data = {}
	word2idx = {}
	for line in fin:
		tokens = line.rstrip().split(' ')
		if tokens[0] in vocab:
			data[tokens[0]] = np.array(tokens[1:], dtype=np.float32)
			word2idx[tokens[0]] = len(word2idx)
	word2idx['<unk>'] = len(word2idx)
	word2idx['<pad>'] = len(word2idx)

	return data, word2idx

def vectorize_word2idx(x, word2idx, max_len):
	sent = [word2idx[y] if y in word2idx else word2idx['<unk>'] for y in x.split(' ')]
	while len(sent) < max_len:
		sent.append(word2idx['<pad>'])
	return sent

def get_data(language, emotion, device):
	class FTDataset(Dataset):
		def __init__(self, df, max_len, word2idx=None):
			self.X = df['text'].apply(lambda x: vectorize_word2idx(x, word2idx, max_len))
			self.y = df['label']

		def __len__(self):
			return len(self.y)

		def __getitem__(self, idx):
			X = self.X[idx]
			y = self.y[idx]
			return X, y

	embedding_dim = 300
	if language == 'en' or language == 'en_es' or language == 'en_ar':
		language_tag = 'En'

	if language == 'es':
		language_tag = 'Es'

	if language == 'ar':
		language_tag = 'Ar'

	if emotion != 'valence':
		tr = open('data/2018-EI-reg-{}-{}-train.txt'.format(language_tag, emotion), encoding="utf8")
		te = open('data/2018-EI-reg-{}-{}-test-gold.txt'.format(language_tag, emotion), encoding="utf8")
		dv = open('data/2018-EI-reg-{}-{}-dev.txt'.format(language_tag, emotion), encoding="utf8")
	else:
		tr = open('data/2018-Valence-reg-{}-train.txt'.format(language_tag), encoding="utf8")
		te = open('data/2018-Valence-reg-{}-test-gold.txt'.format(language_tag), encoding="utf8")
		dv = open('data/2018-Valence-reg-{}-dev.txt'.format(language_tag), encoding="utf8")

	# Training examples
	vocab = set()

	max_len=0

	tr_ex = []
	for line in tr:
		l = line.split('\t')
		sent = clean(l[1])
		vocab.update(sent.split(' '))
		max_len = max(max_len, len(sent.split(' ')))

		tr_ex.append((sent,l[-1][:-2]))
	tr_ex = tr_ex[1:]

	# Test examples
	te_ex = []
	for line in te:
		l = line.split('\t')
		sent = clean(l[1])
		max_len = max(max_len, len(sent.split(' ')))

		te_ex.append((sent,l[-1][:-2]))
	te_ex = te_ex[1:]

	# Dev (validation) examples
	dv_ex = []
	for line in dv:
		l = line.split('\t')
		sent = clean(l[1])
		max_len = max(max_len, len(sent.split(' ')))

		dv_ex.append((sent,l[-1][:-2]))
	dv_ex = dv_ex[1:]

	# Format examples into lists
	te_x, te_y = [te_ex[i][0] for i in range(len(te_ex))], [float(te_ex[i][1]) for i in range(len(te_ex))] 
	dv_x, dv_y = [dv_ex[i][0] for i in range(len(dv_ex))], [float(dv_ex[i][1]) for i in range(len(dv_ex))] 
	tr_x, tr_y = [tr_ex[i][0] for i in range(len(tr_ex))], [float(tr_ex[i][1]) for i in range(len(tr_ex))]

	#tr_x, tr_y, dv_x, dv_y, te_x, te_y = tr_x.to(device), tr_y.to(device), dv_x.to(device), dv_y.to(device), te_x.to(device), te_y.to(device)

	if language == 'en' or language == 'en_es' or language == 'en_ar':
		embeddings_matrix, word2idx = load_vectors('ft_en', vocab)

	if language == 'es':
		embeddings_matrix, word2idx = load_vectors('ft_es', vocab)

	if language == 'ar':
		embeddings_matrix, word2idx = load_vectors('ft_ar', vocab)


	tr_dataset = FTDataset(pd.DataFrame(list(zip(tr_x, tr_y)), columns=['text', 'label']), max_len, word2idx)
	dv_dataset = FTDataset(pd.DataFrame(list(zip(dv_x, dv_y)), columns=['text', 'label']), max_len ,word2idx)
	te_dataset = FTDataset(pd.DataFrame(list(zip(te_x, te_y)), columns=['text', 'label']), max_len, word2idx)

	# Put datasets into loaders
	train_loader = torch.utils.data.DataLoader(tr_dataset, batch_size=24, shuffle=True, drop_last=True)#, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
	dev_loader = torch.utils.data.DataLoader(dv_dataset, batch_size=24, shuffle=True, drop_last=True)#, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
	test_loader = torch.utils.data.DataLoader(te_dataset, batch_size=24, shuffle=True, drop_last=True)#, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))

	return train_loader, dev_loader, test_loader, embeddings_matrix, word2idx, tr_x, tr_y, dv_x, dv_y, te_x, te_y, max_len


def get_model(embeddings_matrix, word2idx, hid_size=128):

	#use this
	class fastTextRegressor(nn.Module):
		def __init__(self, embeddings, embedding_dim, hid_size):
			super(fastTextRegressor, self).__init__()

			self.embeddings = nn.Embedding.from_pretrained(embeddings=embeddings)
			self.linear1 = nn.Linear(embedding_dim, 1)
			#self.activation = nn.ReLU()
			#self.linear2 = nn.Linear(hid_size , 1, dtype=torch.float64)

		def forward(self, x):
			if len(x.shape) == 1:
				out = self.embeddings(x).permute(1, 0)
				out = out.sum(dim=1) #pooling = sum from Pytorch
				out = self.linear1(out) #cast to float, pass through linear 1st layer
				return out
			
			out = self.embeddings(x).permute(1, 0, 2)
			out = out.sum(dim=1) #pooling = sum from Pytorch
			out = self.linear1(out) #cast to float, pass through linear 1st layer
			return out

	# Convert embeddings to a PyTorch tensor
	
	embeddings = np.zeros((len(word2idx), 300))
	for word, i in word2idx.items():
		if word == '<unk>':
			embeddings[i] = np.zeros(300)
		elif word == '<pad>':
			embeddings[i] = np.random.randn(300)
		else:
			embeddings[i] = embeddings_matrix[word] 
	
	# Convert embeddings to a PyTorch tensor
	embeddings = torch.tensor(embeddings, dtype=torch.float32)
	
	model = fastTextRegressor(embeddings, 300, hid_size)
	return model


def get_results(tr_x, te_y, dv_y, train_pred, test_pred, dev_pred):
	# Store results of train_pred
	results['MSE'][language][emotion]['train'] = mean_squared_error(tr_y, train_pred)
	results['Pearson'][language][emotion]['train'] = pearsonr(train_pred, tr_y)[0]
	results['Spearman'][language][emotion]['train'] = spearmanr(train_pred, tr_y)[0]


	# Store results of dev_pred
	results['MSE'][language][emotion]['dev'] = mean_squared_error(dv_y, dev_pred)
	results['Pearson'][language][emotion]['dev'] = pearsonr(dev_pred, dv_y)[0]
	results['Spearman'][language][emotion]['dev'] = spearmanr(dev_pred, dv_y)[0]


	# Store results of test_pred
	results['MSE'][language][emotion]['test'] = mean_squared_error(te_y, test_pred)
	results['Pearson'][language][emotion]['test'] = pearsonr(test_pred, te_y)[0]
	results['Spearman'][language][emotion]['test'] = spearmanr(test_pred, te_y)[0]
	return results

def write_results(results, eec_preds, model, freeze):
	# Save results as JSON
	with open("results_ft.json".format(model, freeze),"w") as f:
		json.dump(results,f)
	with open("eec_preds_ft.json".format(model, freeze),"w") as f:
		json.dump(eec_preds, f)

def print_results(language, emotion, results, eec_preds, model, freeze):
	# Save results as JSON
	with open("results_{}_{}_ft.json".format(language, emotion),"w") as f:
		json.dump(results,f)
	with open("eec_preds_{}_{}_ft.json".format(language, emotion),"w") as f:
		json.dump(eec_preds, f)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	#bert, mbert, xlmroberta
	parser.add_argument('--model', type=str, default='bert')
	parser.add_argument('--freeze', action='store_true', help='include flag to freeze params')

	args = parser.parse_args()

	print('***** MLI-FAIRNESS *****')
	
	# Results dictionary
	results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
	# EEC predictions dictionary
	eec_preds = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
	# List of languages
	languages = ['en', 'en_es', 'en_ar', 'es', 'ar']
	# List of emotions (also our datasets)
	emotions = ['anger', 'fear', 'joy', 'sadness', 'valence']

	# Training Variables
	epochs = 20
	learning_rate = 0.001
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print('DEVICE:', device)

	# Loop through each (language, emotion/dataset) pair
	for language in languages:
		for emotion in emotions:

			print('***', language, emotion, '***')



			#eec_dict_cur = eec_dict[language][emotion]
			eec_dict_cur = get_eec_dict(language, emotion)

			#tr_x, tr_y, dv_x, dv_y, te_x, te_y = get_data(language, emotion)
			#train_loader, dev_loader, test_loader = get_dataloaders(tr_x, tr_y, dv_x, dv_y, te_x, te_y)

			train_loader, dev_loader, test_loader, embeddings, word2idx, tr_x, tr_y, dv_x, dv_y, te_x, te_y, max_len = get_data(language, emotion, device)


			model = get_model(embeddings, word2idx)
			model.to(device)

			# Train the model
			loss_fn = torch.nn.MSELoss()
			optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

			# Loop through epochs
			prev_loss = math.inf
			for epoch in range(epochs):

				# Keep track of statistics
				total_loss = 0
				running_loss = 0
				cnt = 0

				for example in train_loader:

					# Get inputs and labels
					inputs = example[0]
					labels = example[1]

					# Zero the parameter gradients
					optimizer.zero_grad()

					inputs = torch.stack((inputs))

					inputs = inputs.to(device)
					labels = labels.to(device)

					outputs = model(inputs).squeeze()

					loss = loss_fn(outputs, labels.float())
					loss.backward()
					optimizer.step()
					cnt = cnt + 1
					running_loss += loss.item()

				for example in dev_loader:
					inputs = example[0]
					labels = example[1]

					inputs = torch.stack((inputs))

					inputs = inputs.to(device)
					labels = labels.to(device)

					outputs = model(inputs).squeeze()


					dev_loss = loss_fn(outputs, labels.float())

					total_loss += dev_loss.item()
					print(total_loss)
					if prev_loss - total_loss < 0.005 and epoch > 10: 
						print("EARLY STOPPING")
						print("EPOCH #")
						print(epoch)
						break
					else:
						prev_loss = total_loss

				# Print statistics
				print('ESTIMATED LOSS:', running_loss/cnt)
				
			print('ok finished training', language, emotion)

			model.to('cpu')

			# Create train_pred
			train_pred = []
			with torch.no_grad():
				for x in tr_x:
					sent = [word2idx[y] if y in word2idx else word2idx['<unk>'] for y in x.split(' ')]
					while len(sent) < max_len:
						sent.append(word2idx['<pad>'])
					sent = torch.tensor(sent).to(device)

					train_pred.append(model(sent).squeeze().item())

			# Create dev_pred
			dev_pred = []
			with torch.no_grad():
				for x in dv_x:
					sent = [word2idx[y] if y in word2idx else word2idx['<unk>'] for y in x.split(' ')]
					while len(sent) < max_len:
						sent.append(word2idx['<pad>'])
					sent = torch.tensor(sent).to(device)
					dev_pred.append(model(sent).squeeze().item())

			# Create test_pred
			test_pred = []
			with torch.no_grad():
				for x in te_x:

					sent = [word2idx[y] if y in word2idx else word2idx['<unk>'] for y in x.split(' ')]
					while len(sent) < max_len:
						sent.append(word2idx['<pad>'])
					sent = torch.tensor(sent).to(device)
					test_pred.append(model(sent).squeeze().item())

			#Create EEC preds
			for k, v in eec_dict_cur.items():
				for x in v:

					sent = [word2idx[y] if y in word2idx else word2idx['<unk>'] for y in x.split(' ')]
					while len(sent) < max_len:
						sent.append(word2idx['<pad>'])
					sent = torch.tensor(sent).to(device)
					eec_preds[language][emotion][k].append(model(sent).squeeze().item())


			results = get_results(tr_x, te_y, dv_y, train_pred, test_pred, dev_pred)

			print_results(language, emotion, results, eec_preds[language][emotion], args.model, args.freeze)

	write_results(results, eec_preds, args.model, args.freeze)