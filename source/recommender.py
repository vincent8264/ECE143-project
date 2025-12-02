import numpy as np
import pandas as pd

from source.similarity import CosineSimilarity


class Recommender:
    def __init__(self, df):
        df = df.drop_duplicates(subset=["track_id"], keep="first")  # Having the same track id means the same song
        df = df.drop_duplicates(subset=["artists", "track_name"],
                                keep="first")  # Also remove songs the have same artist, track name, possibly duplicates too

        df = df.reset_index(drop=True)
        # print(len(df))
        used_features = ["track_id", "artists", "track_name", "acousticness", "danceability", "energy", "key",
                         "speechiness", "loudness", "instrumentalness", "tempo", "track_genre", "valence"]
        df = df[used_features]

        genre_cat = df["track_genre"].astype("category")
        num_genres = len(genre_cat.cat.categories)

        def one_hot(idx, num_classes):
            v = np.zeros(num_classes, dtype=float)
            v[idx] = 1.0
            return v

        df["track_genre"] = genre_cat.cat.codes.apply(
            lambda i: one_hot(i, num_genres)
        )

        # Normalize
        for col in ["danceability", "energy", "key", "speechiness", "instrumentalness", "tempo", "acousticness",
                    "loudness", "valence"]:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
        self.df = df

    def get_track_id(self, artists, track_name):
        """
        Get the id of a song given its album name and track name.
        """
        return self.df[(self.df["artists"] == artists) & (self.df["track_name"] == track_name)]["track_id"].values[0]

    def FlattenVector(self, v):
        """
        Flatten a mixed vector where some elements may be arrays/lists.
        Returns a 1D numpy array of floats.
        """
        flat = []
        for x in v:
            if isinstance(x, (list, np.ndarray)):
                flat.extend(x)
            else:
                flat.append(x)
        return np.array(flat, dtype=float)

    def recommend(self, album_name, track_name, k=5):
        """
        Recommend songs similar to the given track.

        Steps:
        1. Precompute all song vectors (for faster calculation)
        2. Get the input song's vector
        3. Compute distances to all other songs
        4. Return the top-k similiar songs

        """

        mask = ["danceability", "energy", "key", "speechiness", "instrumentalness", "tempo", "track_genre",
                "acousticness", "loudness", "valence"]

        # 1. prepare feature matrix
        # apply FlattenVector to all rows to create feature matrix
        feature_matrix = np.stack(self.df[mask].apply(lambda row: self.FlattenVector(row), axis=1))

        # 2: get the input song's vector
        track_id = self.get_track_id(album_name, track_name)
        target_index = self.df[self.df["track_id"] == track_id].index[0]
        target_vec = feature_matrix[target_index]

        # 3. compute similarity scores of the every song vs target
        similarity_scores = CosineSimilarity(feature_matrix, target_vec)

        # feature "artists" is very hard to handle (one-hot could cause dimensional explode!), so I suggest just set as bonus one
        target_artists = self.df.loc[target_index, "artists"]
        artists_match_mask = (self.df["artists"] == target_artists).values
        similarity_scores[artists_match_mask] *= 1

        # 4. get closest songs
        similarity_scores[target_index] = -1  # Exclude the track itself
        top_k_indices = np.argsort(similarity_scores)[::-1][:k]

        return self.df.iloc[top_k_indices]
