import os
import pandas as pd
import numpy as np

columns = ['none', 'ros', 'smote', 'borderline', 'adasyn', 'smote-enn', 'smote-tomek']

def consolidate_all(consolidated_results_dir):
    final_directory = consolidated_results_dir + '\\all'

    if not os.path.isdir(final_directory):
        os.mkdir(final_directory)

    # List all the files in the current directory
    files = os.listdir(consolidated_results_dir)

    final_data_frame = pd.DataFrame(columns=columns)
    for file in files:
        # For each  classification algorithm, we need to check each of the classes
        file_path = consolidated_results_dir + '\\' + file
        print('File being processed: '+file_path)

        if file != 'all':
            data_frame = pd.read_csv(file_path, sep=';')
            print('Opening DataFrame...')

            result = data_frame.iloc[:,1:]
            df_size = result.shape
            class_array = np.full(shape=(df_size[0]), fill_value=str(file[:-4]))

            result['classifier'] = class_array
            final_data_frame = final_data_frame.append(result, ignore_index=True)


    final_file_path = final_directory + '\\AllResultsResampling.csv'

    final_data_frame.to_csv(final_file_path, sep=';')

    ## Logic to create the file to compare each classification algorithm
    unique_classification_algorithms = np.unique(final_data_frame['classifier'])
    classifier_comparison = pd.DataFrame(columns=unique_classification_algorithms)

    for resampling in columns:
        temp_df = pd.DataFrame(columns=unique_classification_algorithms)
        for algorithm in unique_classification_algorithms:
            sub_data_frame = final_data_frame[final_data_frame['classifier']== algorithm]
            column_to_be_updated = sub_data_frame[resampling]
            temp_df[algorithm] = column_to_be_updated.to_numpy()
            temp_df['resampling_algorithm'] = resampling
        classifier_comparison = classifier_comparison.append(temp_df, ignore_index=True)

    classifier_comparison.to_csv(final_directory + '\\AllResultsClassifiers.csv', sep=';')
    print('Finished parsing')

