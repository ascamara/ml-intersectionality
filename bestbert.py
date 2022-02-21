import numpy as np
import pandas as pd
import torch
import json
import argparse
import transformers # pytorch transformers
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoConfig
import math
from torch.utils.data.dataloader import default_collate

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

	return text

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

def get_hidden_size(model, tokenizer):
	"""
	- Find the size of a hidden layer in the model
	@param   model: a BERT model.
	@param   tokenizer: a BERT tokenizer.
	@return	 size (int): the size of the hidden layer in the model.
	"""
	with torch.no_grad():
		x = tokenizer('Sample sentence for tokenizer', padding='max_length', max_length=128, truncation=True, return_tensors='pt')
		size = model(**x)[0][:,0,:].shape[1]
	return size

def get_models(model, emotions, freeze):

	finetuned_model_dict = {}

	# The actual model that we will use to predict sentiment
	class ModelRegressor(torch.nn.Module): 
		def __init__(self, model_class, tokenizer, pretrained_weights, freeze): 
			super(ModelRegressor, self).__init__()
			# BERT
			self.model_class = model_class
			self.tokenizer = tokenizer
			self.pretrained_weights = pretrained_weights
			self.bert = self.model_class.from_pretrained(self.pretrained_weights)
			# Uncomment next 2 lines to freeze BERT
			if freeze:
				for p in self.bert.parameters():
					p.requires_grad = False
			# Feedforward layers
			self.fc0 = torch.nn.Linear(get_hidden_size(self.bert, self.tokenizer), 128)
			self.sm0 = torch.nn.ReLU()
			self.fc1 = torch.nn.Linear(128, 1)

		def forward(self, x):
			# Tokenize, without gradient
			with torch.no_grad():
				x = self.tokenizer(x, padding='max_length', max_length=128, truncation=True, return_tensors='pt')
			# Feed tokens into BERT
			net = self.bert(**x)[0][:,0,:]
			net = torch.flatten(net, start_dim=1)
			# Feed BERT into feed forward net
			net = self.sm0(self.fc0(net))
			net = self.fc1(net)
			return net

	if model == 'bert':
		
		tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
		model_class = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
		pretrained_weights = "bert-base-uncased"

		finetuned_model_dict['en'] = {emotion : ModelRegressor(model_class, tokenizer, pretrained_weights, freeze) for emotion in emotions}
		finetuned_model_dict['en_es'] = {emotion : ModelRegressor(model_class, tokenizer, pretrained_weights, freeze) for emotion in emotions}
		finetuned_model_dict['en_ar'] = {emotion : ModelRegressor(model_class, tokenizer, pretrained_weights, freeze) for emotion in emotions}

		tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
		model_class = AutoModelForMaskedLM.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
		pretrained_weights = "dccuchile/bert-base-spanish-wwm-uncased"

		finetuned_model_dict['es'] = {emotion : ModelRegressor(model_class, tokenizer, pretrained_weights, freeze) for emotion in emotions}

		tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
		model_class = AutoModelForMaskedLM.from_pretrained("asafaya/bert-base-arabic")
		pretrained_weights = "asafaya/bert-base-arabic"

		finetuned_model_dict['ar'] = {emotion : ModelRegressor(model_class, tokenizer, pretrained_weights, freeze) for emotion in emotions}


	if model == 'mbert':

		pretrained_weights = "bert-base-multilingual-cased"
		tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
		model_class = AutoModelForMaskedLM.from_pretrained("bert-base-multilingual-cased")

		finetuned_model_dict = {language : {emotion : ModelRegressor(model_class, tokenizer, pretrained_weights, freeze) for emotion in emotions} for language in languages}


	if model == 'xlmroberta':

		pretrained_weights = "xlm-roberta-base"
		tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
		model_class = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

		finetuned_model_dict = {language : {emotion : ModelRegressor(model_class, tokenizer, pretrained_weights, freeze) for emotion in emotions} for language in languages}

	return finetuned_model_dict


def get_data(language, emotion, device):
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
	tr_ex = []
	for line in tr:
		l = line.split('\t')
		tr_ex.append((clean(l[1]),l[-1][:-2]))
	tr_ex = tr_ex[1:]

	# Test examples
	te_ex = []
	for line in te:
		l = line.split('\t')
		te_ex.append((clean(l[1]),l[-1][:-2]))
	te_ex = te_ex[1:]

	# Dev (validation) examples
	dv_ex = []
	for line in dv:
		l = line.split('\t')
		dv_ex.append((clean(l[1]),l[-1][:-2]))
	dv_ex = dv_ex[1:]

	# Format examples into lists
	te_x, te_y = [te_ex[i][0] for i in range(len(te_ex))], [float(te_ex[i][1]) for i in range(len(te_ex))] 
	dv_x, dv_y = [dv_ex[i][0] for i in range(len(dv_ex))], [float(dv_ex[i][1]) for i in range(len(dv_ex))] 
	tr_x, tr_y = [tr_ex[i][0] for i in range(len(tr_ex))], [float(tr_ex[i][1]) for i in range(len(tr_ex))]

	return tr_x.to(device), tr_y.to(device), dv_x.to(device), dv_y.to(device), te_x.to(device), te_y.to(device)

def get_dataloaders(tr_x, tr_y, dv_x, dv_y, te_x, te_y):
	# Put x and y into Dataset objects
	train_dataset = Dataset.from_pandas(pd.DataFrame(list(zip(tr_x, tr_y)), columns=['text', 'label']))
	dev_dataset = Dataset.from_pandas(pd.DataFrame(list(zip(dv_x, dv_y)), columns=['text', 'label']))
	test_dataset = Dataset.from_pandas(pd.DataFrame(list(zip(te_x, te_y)), columns=['text', 'label']))

	# Put datasets into loaders
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
	dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=32, shuffle=True)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

	return train_loader, dev_loader, test_loader

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
	with open("results_{}_{}.json".format(model, freeze),"w") as f:
		json.dump(results,f)
	with open("eec_preds_{}_{}.json".format(model, freeze),"w") as f:
		json.dump(eec_preds, f)

def print_results(language, emotion, results, eec_preds, model, freeze):
	# Save results as JSON
	with open("results_{}_{}_{}_{}.json".format(language, emotion, model, freeze),"w") as f:
		json.dump(results,f)
	with open("eec_preds_{}_{}_{}_{}.json".format(language, emotion, model, freeze),"w") as f:
		json.dump(eec_preds, f)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	#bert, mbert, xlmroberta
	parser.add_argument('--model', type=str, default='bert')
	parser.add_argument('--freeze', action='store_true', help='include flag to freeze params')

	args = parser.parse_args()

	print('***** MLI-FAIRNESS *****')

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print('DEVICE: ', device)
	print('model: ', args.model)
	print('freeze: ', args.freeze)

	
	# Results dictionary
	results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
	# EEC predictions dictionary
	eec_preds = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
	# List of languages
	languages = ['en', 'en_es', 'en_ar', 'es', 'ar']
	# List of emotions (also our datasets)
	emotions = ['anger', 'fear', 'joy', 'sadness', 'valence']

	eec_dict = get_eecs()
	finetuned_model_dict = get_models(args.model, emotions, args.freeze)

	# Training Variables
	epochs = 20
	learning_rate = 0.001

	# Loop through each (language, emotion/dataset) pair
	for language in languages:
		for emotion in emotions:

			print('***', language, emotion, '***')

			eec_dict_cur = eec_dict[language][emotion]

			# Move model to device
			finetuned_model_dict[language][emotion].to(device)

			tr_x, tr_y, dv_x, dv_y, te_x, te_y = get_data(language, emotion, device)
			train_loader, dev_loader, test_loader = get_dataloaders(tr_x, tr_y, dv_x, dv_y, te_x, te_y)

			# Train the model
			loss_fn = torch.nn.MSELoss()
			optimizer = torch.optim.Adam(finetuned_model_dict[language][emotion].parameters(), lr=learning_rate)

			# Loop through epochs
			prev_loss = math.inf
			for epoch in range(epochs):

				# Keep track of statistics
				running_loss = 0
				cnt = 0

				for i, data in tqdm(enumerate(train_loader, 0)):

					# Get inputs and labels
					inputs = data['text']
					labels = data['label'].float()

					# Zero the parameter gradients
					optimizer.zero_grad()

					# Forward, backward, optimize
					outputs = finetuned_model_dict[language][emotion](inputs).squeeze()
					loss = loss_fn(outputs, labels)
					loss.backward()
					optimizer.step()
					cnt = cnt + 1
					running_loss += loss.item()

				for i, data in tqdm(enumerate(train_loader, 0)):
					inputs = data['text']
					labels = data['label'].float()

					inputs = torch.stack((inputs))


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


			# Create train_pred
			train_pred = []
			with torch.no_grad():
				for x in tr_x:
					train_pred.append(finetuned_model_dict[language][emotion](x).item())

			# Create dev_pred
			dev_pred = []
			with torch.no_grad():
				for x in dv_x:
					dev_pred.append(finetuned_model_dict[language][emotion](x).item())

			# Create test_pred
			test_pred = []
			with torch.no_grad():
				for x in te_x:
					test_pred.append(finetuned_model_dict[language][emotion](x).item())

			# Move device back to CPU
			finetuned_model_dict[language][emotion].to('cpu')

			#Create EEC preds
			for k, v in eec_dict_cur.items():
				for x in v:
					eec_preds[language][emotion][k].append(finetuned_model_dict[language][emotion](x).item())


			results = get_results(tr_x, te_y, dv_y, train_pred, test_pred, dev_pred)

			print_results(language, emotion, results, eec_preds[language][emotion], args.model, args.freeze)

	write_results(results, eec_preds, args.model, args.freeze)