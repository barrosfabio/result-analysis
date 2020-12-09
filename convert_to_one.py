import pandas as pd
import os

classifiers = ['vot']

for classifier in classifiers:
    file_directory = 'C:/Users/Fabio Barros/Desktop/Qualificação/Novos Resultados/' + classifier
    columns = ['none', 'ros', 'smote', 'borderline', 'adasyn', 'smote-enn', 'smote-tomek']
    final_df = pd.DataFrame()
    final_file_name = 'C:\\Users\\Fabio Barros\\Desktop\\Qualificação\\Novos Resultados\\consolidated\\results_' + classifier + '.csv'

    dir_list = os.listdir(file_directory)

    for dir in dir_list:
        file_path = file_directory + '\\' + dir + '\\global\\experiment_results.csv'
        print(file_path)
        data_frame = pd.read_csv(file_path, sep=';')
        transposed_data_frame = data_frame.transpose()

        final_df = final_df.append(transposed_data_frame)

    final_df = final_df.filter(like='f1_score', axis=0)
    final_df.to_csv(final_file_name, sep=';')
