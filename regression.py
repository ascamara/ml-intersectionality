import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

from scipy.optimize import minimize

from scipy.special import loggamma
from scipy.special import expit, logit
from scipy.stats import t

#https://towardsdatascience.com/a-guide-to-the-regression-of-rates-and-proportions-bcfe1c35344f
def logLikelihood(params, y, X):
	b = np.array(params[0:-1])      # the beta parameters of the regression model
	phi = params[-1]                # the phi parameter
	mu = expit(np.dot(X,b))

	eps = 1e-6                      # used for safety of the gamma and log functions avoiding inf

	res = - np.sum(loggamma(phi+eps) # the log likelihood
				- loggamma(mu*phi+eps) 
				- loggamma((1-mu)*phi+eps) 
				+ (mu*phi-1)*np.log(y+eps) 
				+ ((1-mu)*phi-1)*np.log(1-y+eps))


	return res

def run_regression(y,X):
	# initial parameters for optimization
	phi = 1
	b0 = 1
	x0 = np.array([b0,b0,b0,phi])

	res = minimize(
		logLikelihood, 
		x0=x0,
		args=(y,X), 
		bounds=[(None,None), (None,None), (None,None), (0,None)])

	b = np.array(res.x[0:X.shape[1]])   # optimal regression parameters
	y_ = expit(np.dot(X,b))

	return y_, b


def get_labels(k):
	r = 0
	g = -1
	i = 0

	if len(k.split(' ')) == 2:
		g, r = k.split(' ')

		if g == 'male':
			g = 0
		if g == 'female':
			g = 1
		if r == 'European' or r == 'Anglo':
			r = 0
		if r == 'African-American' or r == 'Latino' or r == 'Arab':
			r = 1

		#code intersectionality
		if r == 1 and g == 1:
			i = 1
	else:
		if k == 'male':
			g = 0
		if k == 'female':
			g = 1

	return r, g, i

if __name__ == '__main__':

	fields = ['language', 'emotion', 'model', 'coefficients', 't_statistics', 'p_values']
	rows = []

	languages = ['en', 'en_es', 'en_ar', 'es', 'ar']
	emotions = ['anger', 'fear', 'joy', 'sadness', 'valence']
	models = ['bert_True', 
		'mbert_True',
		'xlmroberta_True',
		'bert_False', 
		'mbert_False',
		'xlmroberta_False',
		'ft',
		'baseline_svm',
		'baseline_lr']


	for language in languages:
		for emotion in emotions:
			for model in models:

				print(language, emotion, model)

				with open("results/eec_preds_{}_{}_{}.json".format(language, emotion, model),"r") as f:
					data = json.load(f)

				y = []
				X = []

				for k, v in data.items():
					if len(k.split(' ')) == 2:

						r, g, i = get_labels(k)

						for elt in v:

							y.append(elt)
							X.append([r,g,i])
				X = np.array([np.array(xi) for xi in X])
				y = np.array([yi for yi in y])


				#print(y.shape)
				#print(X.shape)

				y_pred, b = run_regression(y,X)

				n = len(y)
				k = 3

				#https://jianghaochu.github.io/calculating-t-statistic-for-ols-regression-in-python.html
				residual = y - y_pred  # calculate the residual
				sigma_hat = sum(residual ** 2) / (n - k - 1)  # estimate of error term variance
				variance_beta_hat = sigma_hat * np.linalg.inv(np.matmul(X.transpose(), X))

				t_stat = b / np.sqrt(variance_beta_hat.diagonal())

				p_value = 1 - 2 * np.abs(0.5 - np.vectorize(t.cdf)(t_stat, n - k - 1))

				rows.append([language, emotion, model, b.tolist(), t_stat.tolist(), p_value.tolist()])

with open('statistics_without_intercept.csv', 'w') as f:
      
    # using csv.writer method from CSV package
    write = csv.writer(f)
      
    write.writerow(fields)
    write.writerows(rows)
				


				




