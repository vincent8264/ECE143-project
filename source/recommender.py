import numpy as np

from source.dataset import Dataset


class Recommender:
    def __init__(self, df):
        self.data = Dataset(df)

    def recommend(self, songs=None, k=5,
                  weights=None, alpha=0.5,
                  random=False, top_n=5, seed=0):
        """Recommends k songs similar to a set of input songs."""
        if songs is None:
            print("Recommending most popular songs: ")
            return [(row.artists, row.track_name) for i, row in
                    self.data.df.sort_values(by="popularity", ascending=False).head(k).iterrows()]

        assert isinstance(songs, list)
        track_ids = [self.data.get_track_id(artists, track_name) for artists, track_name in songs]
        track_ids = [track_id for track_id in track_ids if track_id is not None]
        if len(track_ids) == 0:
            print("No songs matched in dataset. Recommending most popular songs: ")
            return [(row.artists, row.track_name) for i, row in
                    self.data.df.sort_values(by="popularity", ascending=False).head(k).iterrows()]

        assert all(len(song) == 2 for song in songs)
        assert isinstance(k, int) and k > 0

        if weights is None:
            weights = {c: 1 for c in self.data.features}
        else:
            assert isinstance(weights, dict)
            assert set(weights.keys()).issubset(self.data.features)
            assert all(isinstance(weight, (int, float)) for weight in weights.values())
            for feature in self.data.features:
                weights.setdefault(feature, 0)

        # 1. feature matrix for cosine similarity
        x = self.data.df[self.data.numerical_features].to_numpy()
        mu = x.mean(axis=0)
        sd = x.std(axis=0)
        sd[sd == 0] = 1
        x = (x - mu) / sd

        # 2: input songs' matrix for cosine similarity
        y_i = self.data.df.index.get_indexer(track_ids)
        y = x[y_i]

        # weights for cosine similarity
        w = [weights[feature] for feature in self.data.numerical_features]

        # 3. compute similarity scores
        cosine_similarity = self.cosine_similarity(x, y, w)
        # jaccard_similarity = self.jaccard_similarity(self.data.df, self.data.df.iloc[], weights)
        # similarity = alpha * cosine_similarity + (1 - alpha) * jaccard_similarity

        cosine_similarity = cosine_similarity.max(axis=1)

        for i in y_i:
            cosine_similarity[i] = -1  # Exclude the track itself

        # 4. get closest songs
        if random:
            np.random.seed(seed)
            top_n_indices = np.argsort(cosine_similarity)[::-1][:top_n]
            top_k_i = np.random.choice(top_n_indices, size=min(k, top_n), replace=False)
        else:
            top_k_i = np.argsort(cosine_similarity)[::-1][:k]

        return [(row.artists, row.track_name) for row in self.data.df.iloc[top_k_i].itertuples()]

    def cosine_similarity(self, x, y, w=None):
        """Computes cosine similarity with optional per-feature weights."""
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
        assert x.shape[1] == y.shape[1]

        if w is not None:
            assert isinstance(w, list) or isinstance(w, np.ndarray)
            w = np.asarray(w)
            assert w.shape[0] == x.shape[1]

            w = np.sqrt(w)
            x = x * w
            y = y * w

        numer = np.dot(x, y.T)
        denom = np.linalg.norm(x, axis=1, keepdims=True) * np.linalg.norm(y, axis=1, keepdims=True)

        return np.divide(numer, denom, out=np.zeros_like(numer, dtype=float), where=denom != 0)

    def jaccard_similarity(self, x, y, weights):
        single_cols = self.data.categorical_features
        tuple_cols = self.data.multiclass_features

        # Initialize similarity accumulator
        sim_acc = np.zeros((x.shape[0], y.shape[0]), dtype=float)

        for col in single_cols:
            sim_acc += self.vectorized_single_match(x, y, col).astype(float)

        # Tuple columns
        for col in tuple_cols:
            sim_acc += self.vectorized_tuple_match(x, y, col).astype(float)

        return sim_acc / (len(tuple_cols) + len(single_cols))

    def vectorized_tuple_match(self, df1, df2, col):
        """
        Returns a boolean matrix: df1 rows vs df2 rows
        True if tuple column has at least one common element
        """
        # Convert tuples to sets
        sets1 = df1[col].apply(set).values
        sets2 = df2[col].apply(set).values

        n1, n2 = len(sets1), len(sets2)

        # Create match matrix
        match_matrix = np.zeros((n1, n2), dtype=bool)
        for i in range(n1):
            # Broadcast intersection check
            match_matrix[i, :] = [bool(sets1[i] & s2) for s2 in sets2]
        return match_matrix

    def vectorized_single_match(self, df1, df2, col):
        """
        Returns a boolean matrix: True if values equal
        """
        return df1[col].values[:, None] == df2[col].values[None, :]
