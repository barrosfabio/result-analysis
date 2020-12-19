import pdfplumber
import pandas as pd
import numpy as np
import os

final_columns = ['none','ros','smote','borderline','adasyn','smote-enn','smote-tomek']

def write_df_csv(path, results_df):
    final_results_df = pd.DataFrame(columns=final_columns)
    final_results_df['none'] = results_df['none']
    final_results_df['ros'] = results_df['ros']
    final_results_df['smote'] = results_df['smote']
    final_results_df['borderline'] = results_df['borderline']
    final_results_df['adasyn'] = results_df['adasyn']
    final_results_df['smote-enn'] = results_df['smote-enn']
    final_results_df['smote-tomek'] = results_df['smote-tomek']
    final_results_df.to_csv(path,sep=';')


def extract_from_pdfs():
    algorithms = ['nb']
    per_class_dir = 'C:\\Users\\Fabio Barros\\Desktop\\Qualificação\\Novos Resultados - Sem Rydls\\per_class\\'

    if not os.path.isdir(per_class_dir):
        os.mkdir(per_class_dir)


    for algorithm in algorithms:
        columns = ['adasyn','borderline','none','ros','smote','smote-enn','smote-tomek']

        key_mapping = {'0':'Bacterial_Klebsiella','1':'Bacterial_Legionella',
                       '2':'Bacterial_Streptococcus','3':'Fungal_Pneumocystis','4':'Lipoid','5':'Viral_COVID','6':'Viral_MERS','7':'Viral_SARS','8':'Tuberculosis'}
        rf_directory = 'C:/Users/Fabio Barros/Desktop/Qualificação/Novos Resultados - Sem Rydls/'+ algorithm + '/'
        result_dir =  per_class_dir + algorithm+'\\'

        if not os.path.isdir(result_dir):
            os.mkdir(result_dir)

        experiments = os.listdir(rf_directory)

        for key, value in key_mapping.items():
            results_df = pd.DataFrame(columns=columns)
            for exp in experiments:
                confusion_matrix_dirs = rf_directory + exp + '\\per_pipeline\\confusion_matrix\\'
                resampling_approach = os.listdir(confusion_matrix_dirs)
                row = {}
                for approach in resampling_approach:
                    dirs = confusion_matrix_dirs  + approach
                    resampling_algorithms = os.listdir(dirs)
                    for resampling in resampling_algorithms:

                        path = dirs + '\\'+ resampling + '\\conf_matrix_'+resampling+'_normalized.pdf'
                        print(path)


                        with pdfplumber.open(path) as pdf:
                            first_page = pdf.pages[0]
                            print('Page Width: {}'.format(first_page.width))
                            print('Page Height: {}'.format(first_page.height))
                            cropped_page = first_page.crop((657, 190,(first_page.width-260), (first_page.height-650)),relative=False)
                            text = cropped_page.extract_text()
                            array = np.fromstring(text, sep='\n').reshape(-1,9)
                            data_frame = pd.DataFrame(array)
                            print('Recall for class {}'.format(value))
                            #print(data_frame.iloc[int(key), int(key)])
                            row[resampling] = data_frame.iloc[int(key), int(key)]
                        print(row)
                results_df = results_df.append(row, ignore_index=True)
            write_df_csv(result_dir + value+'.csv', results_df)

