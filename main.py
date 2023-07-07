import nltk
import pandas as pd
import string
import ast
import re
import unidecode

from nltk.corpus import wordnet
# nltk.download('wordnet')
from nltk.corpus import stopwords
# nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
from nltk import word_tokenize
from preprocess_datasets.measures import measures
from preprocess_datasets.actions import cooking_actions

from sklearn.feature_extraction.text import TfidfVectorizer


from contextlib import redirect_stderr
import warnings



def main():
    data = preprocess()
    data['parsed_new'] = data.ingredients.apply(ingredient_parser)
    # print(data.head())
    print(data)
    print(data.take([13]))
    data.to_csv('out.csv')
    # feature_extraction(data)

def preprocess():
    # warnings_list = []
    warnings.filterwarnings("always", append=True)
    with warnings.catch_warnings(record=True) as w:
        df = pd.read_csv('input/newRecipes.csv', on_bad_lines='warn')
        # for warning in w:
        #     warnings_list.append(warning.message)
    # df = pd.read_csv('input/recipes.csv', on_bad_lines='skip')
    # print(warnings_list)
    # print(len(warnings_list))
    return df
def feature_extraction(data):
    recipies = data['parsed_new'].tolist()
    ingredients = [' '.join(r) for r in recipies]
    print(' '.join(ingredients))
    print(ingredients)
    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Compute TF-IDF features
    features = vectorizer.fit_transform(ingredients)

    # Get the feature names
    feature_names = vectorizer.get_feature_names_out()

    # Print the TF-IDF features for each document
    for i, document in enumerate(ingredients):
        print("Document", i+1)
        for j, feature_idx in enumerate(features[i].indices):
            feature_name = feature_names[feature_idx]
            tfidf_score = features[i, feature_idx]
            print(feature_name, ":", tfidf_score)
        print()
    df = pd.DataFrame(features[i])
    print(df)

def ingredient_parser(ingredients):
    # measures and common words (already lemmatized)   
    # Turn ingredient list from string into a list 
    if isinstance(ingredients, list):
       ingredients = ingredients
    else:
       ingredients = ast.literal_eval(ingredients)
    # We first get rid of all the punctuation
    translator = str.maketrans('', '', string.punctuation)
    # initialize nltk's lemmatizer    
    lemmatizer = WordNetLemmatizer()
    ingred_list = []
    for i in ingredients:
        i_keywords = word_tokenize(i)
        # re.split(' |-', i)
        # print(tokens)
        
        # Get rid of words containing non alphabet letters
        i_keywords = [word for word in i_keywords if word.isalpha()]
        # Turn everything to lowercase
        i_keywords = [word.lower() for word in i_keywords]
        # remove accents
        i_keywords = [unidecode.unidecode(word) for word in i_keywords]
        # Lemmatize words so we can compare words to measuring words
        i_keywords = [lemmatizer.lemmatize(word) for word in i_keywords]
        # get rid of stop words
        stop_words = set(stopwords.words('english'))
        i_keywords = [word for word in i_keywords if word not in stop_words]
        # Gets rid of measuring words/phrases, e.g. heaped teaspoon
        i_keywords = [word for word in i_keywords if word not in measures]
        i_keywords = [word for word in i_keywords if word not in cooking_actions]
        
        tagged_words = pos_tag(i_keywords)
        # print(tagged_words)
        # get rid of adverbs, adjectives, verbs, 
        i_keywords = [word for word, tag in tagged_words if tag not in ['RB', 'RBR', 'RBS', 'JJS', 'JJR', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']]
                    #   in ['RB', 'RBR', 'RBS', 'JJS', 'JJR', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']]
        # i.translate(translator)
        # We split up with hyphens as well as spaces
       

        # Get rid of common easy words
        # i_keywords = [word for word in i_keywords if word not in words_to_remove]
        if i_keywords:
           ingred_list.append(' '.join(i_keywords))
    return ingred_list

if __name__ == "__main__":
    main()