import pandas as pd

# takes in a dataframe
# fix recipe_name column by getting rid of " - Allrecipes.com" and " Recipe"
# return a dataframe

def name_fix(dataframe):
    phrase1 = " - Allrecipes.com"
    phrase2 = " Recipe"
    for row_index in range(len(dataframe)):
        current_cell = str(dataframe['recipe_name'][row_index])

        if phrase1 in current_cell:
            current_cell = current_cell.replace(phrase1, "")
        if phrase2 in current_cell:
            current_cell = current_cell.replace(phrase2, "")
        
        dataframe['recipe_name'][row_index] = current_cell

    return dataframe

