import os
import pandas as pd
import numpy as np

columns = ['none', 'ros', 'smote', 'borderline', 'adasyn', 'smote-enn', 'smote-tomek']
results_directory = 'C:\\Users\\Fabio Barros\\Desktop\\Qualificação\\Novos Resultados - Sem Rydls\\consolidated'
final_directory = 'C:\\Users\\Fabio Barros\\Desktop\\Qualificação\\Novos Resultados - Sem Rydls\\consolidated\\all'

def consolidate_all():


    if not os.path.isdir(final_directory):
        os.mkdir(final_directory)

    # List all the files in the current directory
    files = os.listdir(results_directory)

    final_data_frame = pd.DataFrame(columns=columns)
    for file in files:
        # For each  classification algorithm, we need to check each of the classes
        file_path = results_directory + '\\' + file
        print('File being processed: '+file_path)

        if file != 'all':
            data_frame = pd.read_csv(file_path, sep=';')
            print('Opening DataFrame...')

            result = data_frame.iloc[:,1:]
            df_size = result.shape
            class_array = np.full(shape=(df_size[0]), fill_value=str(file[:-4]))

            result['classifier'] = class_array
            final_data_frame = final_data_frame.append(result, ignore_index=True)


    final_file_path = final_directory + '\\AllResults.csv'

    final_data_frame.to_csv(final_file_path, sep=';')

    ## Logic to create the file to compare each resampling algorithm
    unique_classification_algorithms = np.unique(final_data_frame['classifier'])
    resampling_algorithm_comparison = pd.DataFrame(columns=unique_classification_algorithms)

    for resampling in columns:
        temp_df = pd.DataFrame(columns=unique_classification_algorithms)
        for algorithm in unique_classification_algorithms:
            sub_data_frame = final_data_frame[final_data_frame['classifier']== algorithm]
            column_to_be_updated = sub_data_frame[resampling]
            temp_df[algorithm] = column_to_be_updated.to_numpy()
            temp_df['resampling_algorithm'] = resampling
        resampling_algorithm_comparison = resampling_algorithm_comparison.append(temp_df, ignore_index=True)

    resampling_algorithm_comparison.to_csv(final_directory + '\\AllResultsResampling.csv', sep=';')
    print('Finished parsing')

