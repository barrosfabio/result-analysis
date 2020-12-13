import numpy as np
import pandas as pd
import os

import scikit_posthocs as sp
import scipy.stats as stats
from Orange.evaluation import compute_CD, graph_ranks
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('pdf', fonttype=42)
matplotlib.rc('ps', fonttype=42)

def friedman_test(data_frame, analysis_type, algorithm):
	if analysis_type == 'classifier':
		stat, p = stats.friedmanchisquare(data_frame['none'], data_frame.ros, data_frame.smote, data_frame.borderline,
										  data_frame.adasyn, data_frame['smote-enn'], data_frame['smote-tomek'])
	elif analysis_type == 'resampling_algorithm':
		stat, p = stats.friedmanchisquare(data_frame.ada, data_frame.bag, data_frame.dt, data_frame.ext,
										  data_frame.gbdt, data_frame.knn, data_frame.mlp, data_frame.nb,
										  data_frame.rf, data_frame.stk, data_frame.svm, data_frame.vot)


	print('Statistics=%.6f, p=%.6f' % (stat, p))
	# interpret
	alpha = 0.05
	print('Algorithm being analyzed: '+algorithm)
	if p > alpha:
		print('Same distributions (fail to reject H0)')
	else:
		print('Different distributions (reject H0)')
	return p

def calculate_ranks(figure_path, data_frame, algorithm_analyzed):

	data_frame = data_frame.iloc[:,1:-1]
	names = data_frame.columns
	flat_values = data_frame[names].values

	ranks = np.array([stats.rankdata(array) for array in -flat_values])
	# Calculating the average ranks.
	average_ranks = np.mean(ranks, axis=0)
	print('\n'.join('{} average rank: {}'.format(a, r) for a, r in zip(names, average_ranks)))

	cd = compute_CD(average_ranks,
					n=len(flat_values),
					alpha='0.05',
					test='nemenyi'
					)
	print(f'CD: {cd}')



	graph_ranks(average_ranks,
				names=[og_name + '_' + algorithm_analyzed for og_name in names],
				cd=cd,
				textspace=1.5,
				width=6
				)

	cd_data_frame = pd.DataFrame(columns=['algorithm', 'rank'])
	cd_data_frame['algorithm'] = names.values
	cd_data_frame['rank'] = average_ranks


	cd_data_frame.to_csv(figure_path + '\\avg_ranks_' + algorithm_analyzed + '.csv' , sep=';')
	plt.savefig(figure_path + '\\cd_plot_' + algorithm_analyzed + '.pdf', bbox_inches='tight')
	plt.show()


results_file_path = 'C:\\Users\\Fabio Barros\\Desktop\\Qualificação\\Novos Resultados - Sem Rydls\\consolidated\\all\\'
analysis_results_path = 'C:\\Users\\Fabio Barros\\Desktop\\Qualificação\\Novos Resultados - Sem Rydls\\overall_analysis'
classifier_analysis = 'C:\\Users\\Fabio Barros\\Desktop\\Qualificação\\Novos Resultados - Sem Rydls\\overall_analysis\\classifier'
resampling_analysis = 'C:\\Users\\Fabio Barros\\Desktop\\Qualificação\\Novos Resultados - Sem Rydls\\overall_analysis\\resampling'
if not os.path.isdir(analysis_results_path):
	os.mkdir(analysis_results_path)
	os.mkdir(classifier_analysis)
	os.mkdir(resampling_analysis)

## Load the data for the overall results

result_files = os.listdir(results_file_path)

for file in result_files:
	file_path = results_file_path + file
	result_data = pd.read_csv(file_path, sep=';')
	last_column_name = result_data.keys()
	analysis_type = last_column_name[-1]
	analysis_target = np.unique(result_data[analysis_type])

	p_values = []
	algorithms = []

	# Iterate over each algorithm
	for algorithm in analysis_target:
		# Filter what we want to test
		filtered_data = result_data[result_data[analysis_type] == algorithm]

		# Run Friedman Test
		p_value = friedman_test(filtered_data, analysis_type, algorithm)
		p_values.append(p_value)

		# Calculate Overall Ranks and CD plots
		if analysis_type == 'classifier':
			figure_path = classifier_analysis + '\\' + algorithm
		elif analysis_type == 'resampling_algorithm':
			figure_path = resampling_analysis + '\\' + algorithm

		if not os.path.isdir(figure_path):
			os.mkdir(figure_path)

		calculate_ranks(figure_path, filtered_data, algorithm)

	print(p_values)




