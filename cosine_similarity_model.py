from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#recipe is a dict with a 'name' and 'ingredients'
recipes = [
    {"name": "Pancakes", "ingredients": "milk egg flour sugar butter"},
    {"name": "Greek Salad", "ingredients": "tomato cucumber olive oil salt"},
    {"name": "Chicken Stir Fry", "ingredients": "chicken garlic onion ginger salt"}
]

def recommend_recipes(user_ingredients, recipes):
    #Extract the ingredients field for vectorizing
    recipe_ingredients = [recipe["ingredients"] for recipe in recipes]

    # The list of all recipes, including the user's "recipe"
    all_recipes = recipe_ingredients + [user_ingredients]

    # Use a CountVectorizer to transform the recipes into a "bag of words"
    vectorizer = CountVectorizer().fit_transform(all_recipes)
    vectors = vectorizer.toarray()

    # Calculate the cosine similarity between the user's recipe and all other recipes
    csim = cosine_similarity(vectors)
    
    # Get the similarity scores for the user's recipe with all other recipes
    scores = csim[-1][:-1]

    # Sort the scores to find the most similar recipe
    most_similar_recipe = scores.argsort()[-1]

    # Return the name of the most similar recipe
    return recipes[most_similar_recipe]["name"]

def main():
    user_ingredients = "flour sugar butter"
    print(recommend_recipes(user_ingredients, recipes))


if __name__ == "__main__":
    main()
