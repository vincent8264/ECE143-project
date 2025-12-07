import numpy as np

from source.dataset import Dataset


class Recommender:
    def __init__(self, df):
        self.data = Dataset(df)

    def recommend(self, songs: list[tuple[str]] = None, k: int = 5, filters: dict[str, object] = None,
                  weights: dict[str, float] = None, alpha: float = 0.5,
                  random: bool = False, top_n: int = 5, seed: int = None):
        """Recommends k songs similar to a set of input songs."""
        df = self.data.df
        if filters is not None:
            assert isinstance(filters, dict)
            assert set(filters.keys()).issubset(self.data.features)
            for feature, filter in filters.items():
                if feature in self.data.categorical_features:
                    assert isinstance(filter, list)
                    df = df[df[feature].isin(filter)]
                elif feature in self.data.multiclass_features:
                    assert isinstance(filter, list)
                    df = df[df[feature].apply(lambda x: any(f in x for f in filter))]
                else:
                    assert isinstance(filter, dict)
                    for op, v in filter.items():
                        if op == "gt":
                            df = df[df[feature] > v]
                        elif op == "lt":
                            df = df[df[feature] < v]
                        elif op == "gte":
                            df = df[df[feature] >= v]
                        elif op == "lte":
                            df = df[df[feature] <= v]
                        elif op == "eq":
                            df = df[df[feature] == v]
                        else:
                            raise ValueError(f"Unsupported operator {op} for {feature}")

        if songs is None:
            print("Recommending most popular songs: ")
            return [(row.artists, row.track_name) for i, row in
                    df.sort_values(by="popularity", ascending=False).head(k).iterrows()]

        assert isinstance(songs, list)
        track_ids = [self.data.get_track_id(artists, track_name) for artists, track_name in songs]
        track_ids = [track_id for track_id in track_ids if track_id is not None]
        if len(track_ids) == 0:
            print("No songs matched in dataset. Recommending most popular songs: ")
            return [(row.artists, row.track_name) for i, row in
                    df.sort_values(by="popularity", ascending=False).head(k).iterrows()]

        assert all(len(song) == 2 for song in songs)
        assert isinstance(k, int) and k > 0

        if weights is None:
            weights = {c: 1 for c in self.data.features}
        else:
            assert isinstance(weights, dict)
            assert set(weights.keys()).issubset(self.data.features)
            assert all(isinstance(weight, (int, float)) for weight in weights.values())
            for feature in self.data.categorical_features:
                weights.setdefault(feature, 0)
            for feature in self.data.multiclass_features:
                weights.setdefault(feature, 0)
            for feature in self.data.numerical_features:
                weights.setdefault(feature, 1)

        assert isinstance(alpha, float) and 0 <= alpha <= 1
        assert isinstance(random, bool)
        assert isinstance(top_n, int) and top_n >= k
        assert seed is None or isinstance(seed, int)

        # 1. feature matrix for cosine similarity
        x = df[self.data.numerical_features].to_numpy()
        mu = x.mean(axis=0)
        sd = x.std(axis=0)
        sd[sd == 0] = 1
        x = (x - mu) / sd

        # 2: input songs' matrix for cosine similarity
        y_i = df.index.get_indexer(track_ids)
        y = x[y_i]

        # weights for cosine similarity
        w = [weights[feature] for feature in self.data.numerical_features]

        # 3. compute similarity scores
        cosine_similarity = self.cosine_similarity(x, y, w)

        jaccard_similarity = self.jaccard_similarity(df, df.loc[track_ids], weights)
        similarity = alpha * cosine_similarity + (1 - alpha) * jaccard_similarity

        similarity = similarity.max(axis=1)

        for i in y_i:
            similarity[i] = -1  # exclude the given tracks

        # 4. get closest songs
        top_n_i = np.argsort(similarity)[::-1][:top_n]
        if random:
            if seed is not None:
                np.random.seed(seed)
            top_k_i = np.random.choice(top_n_i, size=k, replace=False)
        else:
            top_k_i = top_n_i[:k]

        return [(row.artists, row.track_name) for row in df.iloc[top_k_i].itertuples()]

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
        denom = np.linalg.norm(x, axis=1, keepdims=True) * np.linalg.norm(y, axis=1, keepdims=True).T

        return np.divide(numer, denom, out=np.zeros_like(numer, dtype=float), where=denom != 0)

    def jaccard_similarity(self, x, y, weights):
        categorical_cols = self.data.categorical_features
        multiclass_cols = self.data.multiclass_features

        sim_acc = np.zeros((x.shape[0], y.shape[0]), dtype=float)
        sim_w = 0

        for col in categorical_cols:
            w = weights.get(col)
            sim_acc += w * (x[col].values[:, None] == y[col].values[None, :]).astype(float)
            sim_w += w

        for col in multiclass_cols:
            w = weights.get(col)
            sim_acc += w * self.multiclass_match(x, y, col).astype(float)
            sim_w += w

        return sim_acc / sim_w

    def multiclass_match(self, df1, df2, col):
        """
        Returns a boolean matrix: df1 rows vs df2 rows
        True if tuple column has at least one common element
        """
        # Convert tuples to sets
        sets1 = df1[col].apply(set).values
        sets2 = df2[col].apply(set).values

        n1, n2 = len(sets1), len(sets2)

        match_matrix = np.zeros((n1, n2), dtype=bool)
        for i in range(n1):
            match_matrix[i, :] = [bool(sets1[i] & s2) for s2 in sets2]
        return match_matrix
