import os
import pandas as pd
import numpy as np


columns = ['none', 'ros', 'smote', 'borderline', 'adasyn', 'smote-enn', 'smote-tomek']


def consolidate_each_class(results_directory, key_mapping):
    final_directory = results_directory + 'per_class'
    final_per_class_path = results_directory + '\\consolidated_per_class'

    if not os.path.isdir(final_per_class_path):
        os.mkdir(final_per_class_path)

    # List of all classification algorithms
    dir_list = os.listdir(final_directory)

    # Iterate over each of the classes (xlsx/csv files)
    for key, value in key_mapping.items():
        file_name = value + '.csv'
        final_data_frame = pd.DataFrame(columns=columns)
        for directory in dir_list:
            # For each  classification algorithm, we need to check each of the classes
            classification_path = final_directory + '\\' + directory
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
            print(avg_results)
            print(avg_results['smote'])
            mean = avg_results.iloc[1,:]
            std = avg_results.iloc[0,:]
            partial_result = partial_result.append(mean)
            partial_result = partial_result.append(std)
            class_array = np.append(class_array, ['mean','std'])
            partial_result['classifier'] = class_array

            final_data_frame = final_data_frame.append(partial_result, ignore_index=True)

        #Building the final result path
        final_file_path = final_per_class_path + '\\' + value +'_Resampling.csv'

        final_data_frame.to_csv(final_file_path, sep=';')

        # Create files for classification approach
        ## Logic to create the file to compare each classification algorithm
        unique_classification_algorithms = np.unique(dir_list)
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
