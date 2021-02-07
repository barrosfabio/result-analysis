import pandas as pd
import os

columns = ['none', 'ros', 'smote', 'borderline', 'adasyn', 'smote-enn', 'smote-tomek']

def write_df_csv(path, results_df):
    final_results_df = pd.DataFrame(columns=columns)
    final_results_df['none'] = results_df.iloc[:,0]
    final_results_df['ros'] = results_df.iloc[:,1]
    final_results_df['smote'] = results_df.iloc[:,2]
    final_results_df['borderline'] = results_df.iloc[:,3]
    final_results_df['adasyn'] = results_df.iloc[:,4]
    final_results_df['smote-enn'] = results_df.iloc[:,5]
    final_results_df['smote-tomek'] = results_df.iloc[:,6]
    final_results_df.to_csv(path,sep=';')

def convert_to_one(results_directory, classifiers):
    final_results_directory = results_directory + '\\consolidated\\'

    if not os.path.isdir(final_results_directory):
        os.mkdir(final_results_directory)

    for classifier in classifiers:
        results_folder = results_directory + '\\' +classifier
        results_df = pd.DataFrame(columns=columns)
        final_file_name = final_results_directory + classifier + '.csv'

        dir_list = os.listdir(results_folder)

        for dir in dir_list:
            file_path = results_folder + '\\' + dir + '\\global\\experiment_results.csv'
            print(file_path)
            data_frame = pd.read_csv(file_path, sep=';')
            transposed_data_frame = data_frame.transpose()

            results_df = results_df.append(transposed_data_frame)

        results_df = results_df.filter(like='f1_score', axis=0)
        write_df_csv(final_file_name, results_df)

