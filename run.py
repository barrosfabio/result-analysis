from convert_to_one import convert_to_one
from extract_from_pdf import extract_from_pdfs
from overall_consolidator import consolidate_all
from per_class_consolidator import consolidate_each_class

# This is the script that run all operations to consolidate the final results


# First converting
convert_to_one()

# Extracting from Confusion Matrices pdfs
extract_from_pdfs()

# Consolidating overall results
consolidate_all()

# Consolidating results for each class
consolidate_each_class()