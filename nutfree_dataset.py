import pandas as pd

nuts = [
    'almond',
    'brazil nut',
    'chestnut',
    'coconut',
    'macadamia nut',
    'peanut',
    'pecan',
    'hickory nut',
    'beechnut',
    'acorn',
    'ginkgo nut',
    'lychee nut',
    'shea nut',
    'pili nut',
    'butternut',
    'heartnut',
    'hazelnut',
    'pistachio',
    'pine nut',
    'kola nut',
    'bunya nut',
    'jack nut',
    'coquito nut',
    'monkey puzzle nut',
    'paradise nut',
    'yellowhorn nut',
    'japanese walnut',
    'black walnut',
    'english walnut',
    'cashew',
    'nut',
    'walnut'
]


# takes in a frame, returns a new frame
def nutfree_dataset(frame):
    new_frame = frame.copy()
    for row_index in range(len(new_frame)):
        current_cell = list(new_frame['ingredients'][row_index])
        # for each ingredient in ingredients, check 
        for elem in current_cell:
            if elem in nuts:
                # delete the row
                new_frame.drop(row_index)
                break
    return new_frame