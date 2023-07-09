from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import pandas as pd
import string
import ast
import re
import unidecode
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk import word_tokenize
from input.measures import measures
from input.actions import cooking_actions


def recommend_recipes(user_ingredients, recipes):
    #holds the ingredients for all recipes
    recipe_ingredients = []
    #here we loop over each recipe in recipe list
    for recipe in recipes:
        #here we append the ingredients of the current recipe to recipe_ingredients list
        recipe_ingredients.append(recipe["ingredients"])
    #here we make sure to add the user's ingredients to the end of the recipe_ingredients list
    recipe_ingredients.append(user_ingredients)
    #here we create a CountVectorizer object
    count_vectorizer = CountVectorizer()
    #we use the object to turn the recipe_ingredients list into a "bag of words"
    vectors = count_vectorizer.fit_transform(recipe_ingredients)
    #here we convert the matrix to an array
    vectors_array = vectors.toarray()
    #here we calculate the cosine similarity between all the pairs
    similarity_matrix = cosine_similarity(vectors_array)
    #here we get the number of rows in similarity_matrix which represents the number of recipes
    num_recipes = len(similarity_matrix)
    #here we get the similarity scores for the user's ingredients and the recipes
    similarity_scores = similarity_matrix[num_recipes - 1, :num_recipes - 1]
    #here we get the index of the maximum score
    max_index = 0
    max_score = similarity_scores[0]
    for i in range(1, len(similarity_scores)):
        if similarity_scores[i] > max_score:
            max_score = similarity_scores[i]
            max_index = i
    #we return the most similar recipe
    return recipes[max_index]

def preprocess():
    #here we read in a csv file
    warnings.filterwarnings("always", append=True)
    with warnings.catch_warnings(record=True) as w:
        df = pd.read_csv('input/newRecipes.csv', on_bad_lines='warn')
    return df

def preprocess_ingredients(ingredients):
    #here we use a list of stop words
    stop_words = set(stopwords.words('english'))
    #here we use a list of common measurements and actions
    unnecessary_words = ['cup', 'teaspoon', 'tablespoon', 'lb', 'kg', 'ounce', 'oz', 'grams', 'g', 'liter', 'ml']
    #here we make one big set of all words we don't want to appear
    words_to_remove = stop_words | set(unnecessary_words) | set(measures) | set(cooking_actions)
    processed_ingredients = []
    for ingredient in ingredients:
        #here we convert to lowercase
        ingredient = ingredient.lower()
        #get rid of numbers
        ingredient = re.sub(r'\d+', '', ingredient)
        #we tokenize the words
        words = word_tokenize(ingredient)
        #here we remove stopwords and unneeded words
        words = [word for word in words if word not in words_to_remove]
        #here we join words back to a string and add to the list
        processed_ingredients.append(' '.join(words))
    return processed_ingredients

def remove_phrases(recipe_name):
    #these are the phrases to remove
    phrase1 = " - Allrecipes.com"
    phrase2 = " Recipe"
    #check if they exist and remove them
    if phrase1 in recipe_name:
        recipe_name = recipe_name.replace(phrase1, "")
    if phrase2 in recipe_name:
        recipe_name = recipe_name.replace(phrase2, "")
    return recipe_name

def main():
    data = preprocess()    
    #convert the string list to an actual list
    data['ingredients'] = data['ingredients'].apply(ast.literal_eval)
    #here we preprocess ingridents in the ingredient column
    data['ingredients'] = data['ingredients'].apply(preprocess_ingredients)
    data['recipe_name'] = data['recipe_name'].apply(remove_phrases)
    #here we join the ingredients into a single string for each row
    data['ingredients'] = data['ingredients'].apply(' '.join)
    #here we convert everything into a dictionary
    recipes = data.rename(columns={'RecipeName': 'recipe', 'Link': 'link', 'Ingredients': 'ingredients'}).to_dict('records') 
    user_ingredients = "flour sugar butter"
    print(recommend_recipes(user_ingredients, recipes))


if __name__ == "__main__":
    main()
