import os
import pandas as pd
import numpy as np


columns = ['none', 'ros', 'smote', 'borderline', 'adasyn', 'smote-enn', 'smote-tomek']

results_directory = 'C:\\Users\\Fabio Barros\\Desktop\\Qualificação\\Novos Resultados - Sem Rydls\\per_class'
final_per_class_path = 'C:\\Users\\Fabio Barros\\Desktop\\Qualificação\\Novos Resultados - Sem Rydls\\consolidated_per_class'

def consolidate_each_class():

    if not os.path.isdir(final_per_class_path):
        os.mkdir(final_per_class_path)

    class_list = {'0':'Bacterial_Klebsiella','1':'Bacterial_Legionella',
                  '2':'Bacterial_Streptococcus','3':'Fungal_Pneumocystis','4':'Lipoid','5':'Viral_COVID','6':'Viral_MERS',
                  '7':'Viral_SARS','8':'Tuberculosis'}

    # List of all classification algorithms
    dir_list = os.listdir(results_directory)

    # Iterate over each of the classes (xlsx/csv files)
    for key, value in class_list.items():
        file_name = value + '.csv'
        final_data_frame = pd.DataFrame(columns=columns)
        for directory in dir_list:
            # For each  classification algorithm, we need to check each of the classes
            classification_path = results_directory + '\\' + directory
            class_list = os.listdir(classification_path)
            print('Classifier: '+directory)

            #Building the file path
            file_path = classification_path + '\\' + file_name
            data_frame = pd.read_csv(file_path, sep=';')
            print('Opening DataFrame...')

            partial_result = data_frame.iloc[:,1:]
            df_size = partial_result.shape
            class_array = np.full(shape=(df_size[0]), fill_value=str(directory))

            avg_results = partial_result.describe()
            mean = avg_results['mean']
            std = avg_results['std']
            partial_result = partial_result.append(mean)
            partial_result = partial_result.append(std)
            class_array = class_array.append(['mean','std'])
            partial_result['classifier'] = class_array

            final_data_frame = final_data_frame.append(partial_result, ignore_index=True)

        #Building the final result path
        final_file_path = final_per_class_path + '\\' + value +'_Resampling.csv'

        final_data_frame.to_csv(final_file_path, sep=';')

        # Create files for classification approach
        ## Logic to create the file to compare each classification algorithm
        unique_classification_algorithms = np.unique(final_data_frame['classifier'])
        classifier_comparison = pd.DataFrame(columns=unique_classification_algorithms)

        for resampling in columns:
            temp_df = pd.DataFrame(columns=unique_classification_algorithms)
            for algorithm in unique_classification_algorithms:
                sub_data_frame = final_data_frame[final_data_frame['classifier'] == algorithm]
                column_to_be_updated = sub_data_frame[resampling]
                temp_df[algorithm] = column_to_be_updated.to_numpy()
                temp_df['resampling_algorithm'] = resampling
            classifier_comparison = classifier_comparison.append(temp_df, ignore_index=True)

        classifier_comparison.to_csv(final_per_class_path + '\\' + value + '_Classifier.csv', sep=';')
        print('Finished parsing')
