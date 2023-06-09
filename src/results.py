

from openpyxl import Workbook
from openpyxl.styles import Font
import os
import pandas as pd

if __name__ == "__main__":

    cities = {
        "manaus": "a101",
        "teresina": "a312",
        "rio_janeiro": "a652",
        "brasilia": "a001",
        "curitiba": "a807",
    }

    # Define the folders and their corresponding sheet names
    folders = {
        'temp': 'temp',
        'max_temp': 'max_temp',
        'min_temp': 'min_temp',
        'prep_tot': 'prep_tot',
    }

    # Create a new Excel workbook
    workbook = Workbook()
    path = '../experiments/2023-05-20-00-05'

    # Iterate over each folder
    for folder, sheet_name in folders.items():
        # Create a new sheet in the workbook
        sheet = workbook.create_sheet(title=sheet_name)
        # Get the path to the folder

        for city, code in cities.items():
            file_path = f"{path}/{city}_{code}/results_{folder}.csv"
            df = pd.read_csv(file_path)
            df = df.rename_axis('Model')

            file = os.path.join('path_to_folder_directory', folder)
            sheet.append([city])
            header = ['Model'] + df.columns.tolist()[1:]
            sheet.append(header)  # Write the column headers
            for row in df.itertuples(index=False):
                sheet.append(row)  # Write each row
            sheet.append([])

    # Remove the default sheet created by openpyxl
    workbook.remove(workbook['Sheet'])

    # Save the workbook as an Excel file
    workbook.save('output_results.xlsx')


