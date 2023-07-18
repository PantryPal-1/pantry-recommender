import pandas as pd

non_vegetarian_ingredients = [
    "beef", "chicken", "pork", "lamb", "venison", "rabbit", 
    "turkey", "duck", "goose", "veal", "bison", "boar", 
    "pheasant", "quail", "ostrich", "kangaroo", "frog", 
    "alligator", "goat", "fish", "tuna", "salmon", "cod", 
    "shrimp", "prawn", "lobster", "crab", "oyster", "clam", 
    "mussel", "squid", "octopus", "eel", "caviar", "gelatin", 
    "lard", "tallow", "suet", "stock", "broth", "bone", 
    "rennet", "bone marrow", "liver", "kidney", "heart", "tongue", 
    "anchovy", "sardine", "herring", "mackerel", "swordfish", 
    "halibut", "snapper", "trout", "barramundi", "tilapia", 
    "catfish", "monkfish", "scallop", "cuttlefish", "snail",
    "escargot", "tripe", "sweetbread", "brain", "ham", 
    "bacon", "sausage", "chorizo", "prosciutto", "salami", 
    "pepperoni", "fat", "foie gras", "leg", "belly",
    "roe", "meat", "steak", "rib", "drumstick", "thigh",
    "head", "oxtail", "tail", "pig", "fillet", "neck",
    "ear"]


# takes in a frame, returns a new frame
def veg_dataset(frame):
    new_frame = frame.copy()
    for row_index in range(len(new_frame)):
        current_cell = list(new_frame['ingredients'][row_index])
        # for each ingredient in ingredients, check 
        for elem in current_cell:
            if elem in non_vegetarian_ingredients:
                # delete the row
                new_frame.drop(row_index)
                break
    return new_frame

