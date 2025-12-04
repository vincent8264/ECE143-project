import numpy as np
import pandas as pd

from dataset import Dataset
from source.similarity import CosineSimilarity


class Recommender:
    def __init__(self, df):
        self.data = Dataset(df)

    def recommend(self, songs, k=5, artists_bonus=0.05, random=False, top_n=5, seed=0):
        """
        Recommends k songs similar to a set of input songs.

        Parameters
        ----------
        songs : list of (artists, track_name)
            Example: [("Artists 1", "Song 1"), ("Artists 2", "Song 2")]
        k : int
            Number of recommendations to return.

        Returns
        -------
        pandas.DataFrame
            Top-k recommended tracks.
        """
        assert isinstance(songs, list)
        assert all(len(song) == 2 for song in songs)
        assert isinstance(k, int) and k > 0

        mask = ["danceability", "energy", "key", "speechiness", "instrumentalness", "tempo", "track_genre",
                "acousticness", "loudness", "valence"]

        # 1. prepare feature matrix
        # apply FlattenVector to all rows to create feature matrix
        feature_matrix = np.stack(self.df[mask].apply(lambda row: self.FlattenVector(row), axis=1))

        target_vecs = []
        target_indices = []
        # 2: get the input song's vector
        for artists, track_name in songs:
            track_id = self.get_track_id(artists, track_name)
            if track_id is None:
                continue
            track_i = self.df.index[self.df["track_id"] == track_id][0]
            target_indices.append(track_i)
            target_vecs.append(feature_matrix[track_i])

        # 3. compute similarity scores of the every song vs target
        target_vec = np.mean(target_vecs, axis=0)
        similarity_scores = CosineSimilarity(feature_matrix, target_vec)

        # feature "artists" is very hard to handle (one-hot could cause dimensional explode!), so I suggest just set as bonus one
        input_artists = set(self.df.loc[target_indices, "artists"].values)
        artists_match_mask = self.df["artists"].isin(input_artists).values
        similarity_scores[artists_match_mask] *= 1 + artists_bonus

        # 4. get closest songs
        for target_index in target_indices:
            similarity_scores[target_index] = -1  # Exclude the track itself
        if random:
            np.random.seed(seed)
            top_n_indices = np.argsort(similarity_scores)[::-1][:top_n]
            top_k_indices = np.random.choice(top_n_indices, size=min(k, top_n), replace=False)

        else:
            top_k_indices = np.argsort(similarity_scores)[::-1][:k]

        return [(row.artists, row.track_name) for row in self.df.iloc[top_k_indices].itertuples()]

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
