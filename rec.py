from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class cosine_rec:
    def __init__(self, features, data) -> None:
        self.features = features
        self.data = data

    def predict(self, input, n):
        cosine_sim = cosine_similarity(input, self.features).flatten()
        scores = list(cosine_sim)
        top_results = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        recs = pd.DataFrame(columns=["recipe", "ingredients", "url", "score"])
        for i, ind in enumerate(top_results):
            recs.at[i + 1, "recipe"] = self.data["recipe_name"][ind]
            recs.at[i + 1, "ingredients"] = self.data["ingredients"][ind]
            recs.at[i + 1, "url"] = self.data["recipe_urls"][ind]
            recs.at[i + 1, "score"] = f"{scores[ind]}"
        return recs