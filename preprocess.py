"""File to preprocess data"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#Here we download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')


#here we use a list of English stop words
stop_words = set(stopwords.words('english'))

def preprocess_ingredients(ingredients):
    #here we use a list of common measurements
    unnecessary_words = ['cup', 'teaspoon', 'tablespoon', 'lb', 'kg', 'ounce', 'oz', 'grams', 'g', 'liter', 'ml']
    processed_ingredients = []

    for ingredient in ingredients:
        #here we convert to lowercase
        ingredient = ingredient.lower()

        #Remove numbers
        ingredient = re.sub(r'\d+', '', ingredient)

        #Tokenize the words
        words = word_tokenize(ingredient)

        #Remove stopwords and unneeded words
        words = [word for word in words if word not in stop_words and word not in unnecessary_words]

        #Here we join words back into a string and add to the list
        processed_ingredients.append(' '.join(words))

    return processed_ingredients

ingredients = ["1 cup of milk", "2 teaspoons of salt", "500g of chicken", "1 tablespoon of olive oil"]
print(preprocess_ingredients(ingredients))
