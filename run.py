from convert_to_one import convert_to_one
from extract_from_pdf import extract_from_pdfs
from overall_consolidator import consolidate_all
from per_class_consolidator import consolidate_each_class

# This is the script that run all operations to consolidate the final results
results_directory = "C:\\Users\\Fabio Barros\\Desktop\\IJCNN Novos Resultados\\Rydls + C19 IDC\\"
consolidated_results_directory = results_directory + '\\consolidated'


# class_list = {'0':'Bacterial_Klebsiella','1':'Bacterial_Legionella',
#                   '2':'Bacterial_Streptococcus','3':'Fungal_Pneumocystis','4':'Lipoid','5':'Viral_COVID','6':'Viral_MERS',
#                   '7':'Viral_SARS','8':'Tuberculosis'}

class_list = {'0':'Normals','1':'Bacterial_Klebsiella',
                   '2':'Bacterial_Legionella','3':'Bacterial_Streptococcus','4':'Fungal_Pneumocystis','5':'Lipoid','6':'Viral_COVID',
                   '7':'Viral_MERS','8':'Viral_SARS','9':'Tuberculosis'}




classifiers = ['dt','mlp','svm','knn','nb','gbdt','rf','ext','ada','vot','stk','bag']



# First converting
convert_to_one(results_directory, classifiers)

# Extracting from Confusion Matrices pdfs
extract_from_pdfs(results_directory, class_list, classifiers)

# Consolidating overall results
consolidate_all(consolidated_results_directory)

# Consolidating results for each class
consolidate_each_class(results_directory, class_list)