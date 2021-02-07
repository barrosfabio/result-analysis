import os
import pandas as pd

path = "C:\\Users\\Fabio Barros\\Desktop\\IJCNN Novos Resultados\\Rydls + C19 IDC\\per_class_analysis"
analysis_array = ['resampling', 'classifier']
columns = ['none', 'ros', 'smote', 'borderline', 'adasyn', 'smote-enn', 'smote-tomek']
classifiers = ['ada','bag','dt','ext','gbdt','knn','mlp','nb','rf','stk','svm','vot']


results_df = pd.DataFrame()
temp_data_frame = pd.DataFrame()


def build_final_df_resampling(path, results_df):
    final_results_df = pd.DataFrame(columns=columns)
    final_results_df['none'] = results_df.iloc[:,0]
    final_results_df['ros'] = results_df.iloc[:,1]
    final_results_df['smote'] = results_df.iloc[:,2]
    final_results_df['borderline'] = results_df.iloc[:,3]
    final_results_df['adasyn'] = results_df.iloc[:,4]
    final_results_df['smote-enn'] = results_df.iloc[:,5]
    final_results_df['smote-tomek'] = results_df.iloc[:,6]
    final_results_df['classifier'] = results_df.iloc[:, 7]
    final_results_df['class'] = results_df.iloc[:, 8]
    final_results_df.to_csv(path, sep=';')
    return final_results_df

def build_final_df_classifiers(path, results_df):
    final_results_df = pd.DataFrame(columns=classifiers)
    final_results_df['ada'] = results_df.iloc[:,0]
    final_results_df['bag'] = results_df.iloc[:,1]
    final_results_df['dt'] = results_df.iloc[:,2]
    final_results_df['ext'] = results_df.iloc[:,3]
    final_results_df['gbdt'] = results_df.iloc[:,4]
    final_results_df['knn'] = results_df.iloc[:,5]
    final_results_df['mlp'] = results_df.iloc[:,6]
    final_results_df['nb'] = results_df.iloc[:, 7]
    final_results_df['rf'] = results_df.iloc[:, 8]
    final_results_df['stk'] = results_df.iloc[:, 9]
    final_results_df['svm'] = results_df.iloc[:, 10]
    final_results_df['vot'] = results_df.iloc[:, 11]
    final_results_df['class'] = results_df.iloc[:, 12]
    final_results_df['resampling'] = results_df.iloc[:, 13]
    final_results_df.to_csv(path, sep=';')
    return final_results_df

# Resampling analysis
for analysis in analysis_array:
    per_class_analysis = path + '\\' + analysis
    dir_list = os.listdir(per_class_analysis)
    for directory in dir_list:
        class_path = per_class_analysis + '\\' + directory
        algorithm_list = os.listdir(class_path)
        for algorithm in algorithm_list:
            file_name = 'avg_ranks_' + algorithm + '.csv'
            file_path = class_path + '\\' +algorithm + '\\' + file_name
            local_result = pd.read_csv(file_path, sep=';')
            local_result = local_result.iloc[:,-1]
            local_result = local_result.T
            if analysis == 'resampling':
                local_result['classifier'] = algorithm
            elif analysis == 'classifier':
                local_result['resampling'] = algorithm
            local_result['class'] = directory
            temp_data_frame = temp_data_frame.append(local_result)

    if analysis == 'resampling':
        final_df = build_final_df_resampling(path +'\\resampling_ranks.csv', temp_data_frame)
        temp_data_frame = pd.DataFrame()
    elif analysis == 'classifier':
        final_df = build_final_df_classifiers(path +'\\classifier_ranks.csv', temp_data_frame)

print(final_df)


