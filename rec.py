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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer

def recommend_recipes(user_ingredients, n, data, vegetarian=False):
    """Recommends top n recipes based on user ingredients
    TODO: Experiment with KNN vs Cosine
    returns dataframe of top n recipes
    """
    data['parsed_ingredients'] = data.ingredients.apply(parse_ingredients)
    data['paresed_recipe_name'] = data.recipe_name.apply(parse_recipe_name)
    
    eligible_data = data.copy()  #make copy of the data DataFrame
    
    if vegetarian:  #check for vegetarian flag
        eligible_data = eligible_data[eligible_data['vegetarian'] == True]
    
    input_embedding, features = feature_extraction(user_ingredients, eligible_data)
    cosine_sim = cosine_similarity(input_embedding, features).flatten()
    scores = list(cosine_sim)
    top_results = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    
    recs = pd.DataFrame(columns=["recipe", "ingredients", "url", "score"])
    for i, ind in enumerate(top_results):
        recs.at[i + 1, "recipe"] = eligible_data["recipe_name"][ind]
        recs.at[i + 1, "ingredients"] = eligible_data["ingredients"][ind]
        recs.at[i + 1, "url"] = eligible_data["recipe_urls"][ind]
        recs.at[i + 1, "score"] = f"{scores[ind]}"
    
    return recs


def get_recipe_data():
    """Reads csv data
    returns dataframe of csv data
    """
    warnings.filterwarnings("always", append=True)
    with warnings.catch_warnings(record=True) as w:
        df = pd.read_csv('input/newRecipes.csv', on_bad_lines='warn')
    return df

def parse_ingredients(ingredients):
    """Cleans out scraped ingredients
    TODO: check if POS tagging removes necessary ingredients
    returns ingredient list for a recipe
    """
    if isinstance(ingredients, list):
       ingredients = ingredients
    else:
       ingredients = ast.literal_eval(ingredients)
    stop_words = set(stopwords.words('english'))
    unnecessary_words = ['cup', 'teaspoon', 'tablespoon', 'lb', 'kg', 'ounce', 'oz', 'grams', 'g', 'liter', 'ml']
    words_to_remove = stop_words | set(unnecessary_words) | set(measures) | set(cooking_actions)
    lemmatizer = WordNetLemmatizer()
    processed_ingredients = []
    for ingredient in ingredients:
        ingredient = ingredient.lower()
        words = word_tokenize(ingredient)
        words = [word for word in words if word.isalpha()]
        words = [lemmatizer.lemmatize(word) for word in words] 
        words = [word for word in words if word not in words_to_remove]
        # Need to pos parsing more test more -----------------
        # tagged_words = pos_tag(words)
        # words = [word for word, tag in tagged_words if tag not in ['RB', 'RBR', 'RBS', 'JJS', 'JJR', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']] 
        processed_ingredients.append(' '.join(words))
    return processed_ingredients

def parse_recipe_name(recipe_name):
    """Cleans up a recipe name to only contain relavant ingredient words
    TODO: find words to remove in recipe names
    returns parsed recipe name
    """
    phrase1 = " - Allrecipes.com"
    phrase2 = " Recipe"
    if phrase1 in recipe_name:
        recipe_name = recipe_name.replace(phrase1, "")
    if phrase2 in recipe_name:
        recipe_name = recipe_name.replace(phrase2, "")
    parsed_recipe_name = []
    tokens = word_tokenize(recipe_name)
    stop_words = set(stopwords.words('english'))
    keywords = [word for word in tokens if word not in stop_words]
    parsed_recipe_name.append(' '.join(keywords))
    return parsed_recipe_name

def feature_extraction(user_input, data):
    """Extracts features using TF
    TODO: check if idf is useful
    returns input embedding, features
    """
    recipes = data['parsed_ingredients']
    ingredients = [' '.join(r) for r in recipes]
    recipe_names = data['recipe_name'].tolist()
    input = ' '.join(parse_ingredients(user_input))
    input_df = pd.DataFrame(list(zip([input], [input])), columns=['ingredients', 'recipe_names'])
    i_vectorizer = TfidfVectorizer(ngram_range=(2,2))
    r_vectorizer = TfidfVectorizer(ngram_range=(1,1))
    preprocessor = ColumnTransformer([('i', i_vectorizer, 'ingredients'), ('r', r_vectorizer, 'recipe_names')])
    recipe_data = pd.DataFrame(list(zip(ingredients, recipe_names)), columns=['ingredients', 'recipe_names']) 
    features = preprocessor.fit_transform(recipe_data)
    input_embedding = preprocessor.transform(input_df)
    return input_embedding, features

def vegetarian_recipes(data):
    non_vegetarian_ingredients = ['beef', 'pork', 'chicken', 'fish', 'shrimp', 'tuna', 'steak', 'salmon', 'pepperoni', 'ham', 'salami', 'turkey', 'bacon']
    for index, row in data.iterrows():
        ingredients = row['ingredients']
        #here we look for non-vegetarian items in each recipe
        is_vegetarian = all(ingredient.lower() not in non_vegetarian_ingredients for ingredient in ingredients)
        #we label if recipe is vegetarian or not
        data.at[index, 'vegetarian'] = is_vegetarian
    return data

def main(): 
    user_ingredients = ['rice', 'potatoes']
    data = get_recipe_data()
    data = vegetarian_recipes(data)
    print(recommend_recipes(user_ingredients, 5, data,vegetarian=True))


if __name__ == "__main__":
    main()
