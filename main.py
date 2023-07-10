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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from contextlib import redirect_stderr
import warnings
import numpy as np



def main():
    data = preprocess()
    data['parsed_new'] = data.ingredients.apply(ingredient_parser)
    data['parsed_recipe_name'] = data.recipe_name.apply(recipe_name_parser)
    # print(data)
    # print(data.take([13]))
    # data.to_csv('out.csv')
    df, features, feature_names, scores = feature_extraction(data)
    # sort scores by top 5 best fit
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
    # create dataframe to load in recommendations
    recommendation = pd.DataFrame(columns=["recipe", "ingredients", "score", "url"])
    count = 0
    for i, ind in enumerate(top):
        recommendation.at[i + 1, "recipe"] = data["recipe_name"][ind]
        recommendation.at[i + 1, "ingredients"] = data["ingredients"][ind]
        recommendation.at[i + 1, "url"] = data["recipe_urls"][ind]
        recommendation.at[i, "score"] = f"{scores[ind]}"
        count += 1
    print(recommendation)

    # max_index = 0
    # max_score = scores[0]


    # for i in range(1, len(similarity_scores)):
    #     if similarity_scores[i] > max_score:
    #         max_score = similarity_scores[i]
    #         max_index = i
    # #we return the most similar recipe
    # print(data.loc[max_index])

    # movie_id = df[df['title'] == title]['id'].values[0]
    #print(df)
    #     print(data.loc[i]['recipe_name'])

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

def nmf(features, feature_names, num_clusters):
    nmf = NMF(n_components=num_clusters)
    nmf_features = nmf.fit_transform(features)
    norm_features = normalize(nmf_features)
    components = pd.DataFrame(
        nmf.components_,
    )
    n_top_words = 15
    for i, topic_vec in enumerate(nmf.components_):
        print(i, end=' ')
        for fid in topic_vec.argsort()[-1:-n_top_words-1:-1]:
            print(feature_names[fid], end=' ')
        print()
    return nmf_features

def feature_extraction(data):
    recipes = data['parsed_new'].tolist()
    ingredients = [' '.join(r) for r in recipes]
    # print(' '.join(ingredients))
    # print(ingredients)
    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(ngram_range=(2,2)) # uses unigrams and bigrams

    # Compute TF-IDF features
    # features2 = vectorizer.transform(ingredients)
    features = vectorizer.fit_transform(ingredients)

    # Get the feature names
    feature_names = vectorizer.get_feature_names_out()
    df = pd.DataFrame(
        features.toarray(),
        columns=feature_names,
    )
    # print(df)

    recipe_names = data['recipe_name'].tolist()
    vectorizer = TfidfVectorizer(ngram_range=(1,1))
    vectorizer2 = TfidfVectorizer(ngram_range=(2,2))
    features_r = vectorizer.fit_transform(recipe_names)
    print(features_r)
    
    preprocessor = ColumnTransformer([('i', vectorizer, 'ingredients'), ('r',vectorizer, 'recipe_names')])
    # preprocessor = Pipeline([('recipe_name', vectorizer), ('ingredients', vectorizer2)])
    input = pd.DataFrame(list(zip(ingredients, recipe_names)), columns=['ingredients', 'recipe_names'])
    # print(input)
    res_features = preprocessor.fit_transform(input) 
    training_vectors = preprocessor.transform(input)
    # preprocessor.fit(input)
 
    # Transform the data and format for readibility
    # terms = preprocessor.named_transformers_[
    #     'r'].get_feature_names_out()
    # columns = np.concatenate((['ingredients', 'recipe_names'], terms))
    # # print(res_features)
    # df = pd.DataFrame(
    #     preprocessor.transform(input), columns=columns)

    # print(res_features)
    res_features_names = preprocessor.get_feature_names_out()
    df = pd.DataFrame(res_features.toarray(),columns=res_features_names)


    # Print the TF-IDF features for each document
    # for i, document in enumerate(ingredients):
    #     print("Document", i+1)
    #     for j, feature_idx in enumerate(features[i].indices):
    #         feature_name = feature_names[feature_idx]
    #         tfidf_score = features[i, feature_idx]
    #         print(feature_name, ":", tfidf_score)
    #     print()
    # df = pd.DataFrame(features[i])
    # return(most_similar_recipe_list)

    # create embessing for input text
    input = ['mussels']
    input = ingredient_parser(input)
    # print(input)
    i_string = ' '.join(input)
    # print(i_string)
    input_df = pd.DataFrame(list(zip([i_string], [i_string])), columns=['ingredients', 'recipe_names'])
    # get embeddings for ingredient doc
    print(input_df)
    input_embedding = preprocessor.transform(input_df)
    # print(input_embedding)
    cos_sim = cosine_similarity(input_embedding,res_features).flatten()
    scores = list(cos_sim)
    # print(list(enumerate(scores)))


    return df, features, feature_names, scores

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

def recipe_name_parser(recipe_name):
    parsed_recipes = []
    tokens = word_tokenize(recipe_name)
    stop_words = set(stopwords.words('english'))
    keywords = [word for word in tokens if word not in stop_words]
    if keywords:
        parsed_recipes.append(' '.join(keywords))
    return parsed_recipes
if __name__ == "__main__":
    main()