import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, df):
        assert isinstance(df, pd.DataFrame)
        self.raw = df
        # self.features = ["track_id", "artists", "track_name", "popularity", "danceability", "energy", "key", "loudness",
        #                  "speechiness", "acousticness", "instrumentalness", "valence", "tempo", "track_genre"]
        self.features = ['track_id', 'artists', 'album_name', 'track_name', 'popularity', 'duration_ms', 'explicit',
                         'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                         'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'track_genre']
        assert set(self.features).issubset(df.columns)

        self.categorical_features = ['artists', 'album_name', 'track_name', 'explicit', 'key', 'mode', 'time_signature',
                                     'track_genre']
        self.numerical_features = ['popularity', 'duration_ms', 'danceability', 'energy', 'loudness', 'speechiness',
                                   'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        for c in self.numerical_features:
            assert df[c].dtype == np.int64 or df[c].dtype == np.float64
        self.df = self.clean_data(df)

    def clean_data(self, df):
        df = df.copy()

        # there is one track where artists, album_name, and track_name are None
        df.dropna(subset=['artists', 'track_name'], inplace=True)

        # when multiple rows have the same track_id, the only difference is popularity and track_genre. so average their popularity and combine track_genre
        agg_dict = {}
        for col in self.features:
            if col == 'popularity':
                agg_dict[col] = 'max'
            elif col == 'track_genre':
                agg_dict[col] = lambda x: tuple(sorted(x.unique()))
            else:
                agg_dict[col] = 'first'
        df = df.groupby('track_id', as_index=False).agg(agg_dict)

        df['artists'] = df['artists'].apply(lambda x: tuple(sorted(x.split(';'))))

        # some songs have the same artists and track_name. keep the first one
        df = df.drop_duplicates(subset=["artists", "track_name"], keep="first")

        df = df.reset_index(drop=True)

        # df["tempo"] = pd.to_numeric(df["tempo"], errors="coerce")
        # df = df[df["tempo"] > 0]

        # num_cols = ["danceability", "energy", "key", "speechiness", "instrumentalness", "tempo",
        #             "acousticness", "loudness", "valence"]
        # for c in num_cols:
        #     df[c] = pd.to_numeric(df[c], errors="coerce")
        #     df[c] = df[c].fillna(df[c].median())

        # genre_cat = df["track_genre"].astype("category")
        # num_genres = len(genre_cat.cat.categories)
        #
        # def one_hot(idx, num_classes):
        #     v = np.zeros(num_classes, dtype=float)
        #     v[idx] = 1.0
        #     return v
        #
        # df["track_genre"] = genre_cat.cat.codes.apply(
        #     lambda i: one_hot(i, num_genres)
        # )
        return df

    def normalized_features(self):
        # Normalize
        for c in ["danceability", "energy", "key", "speechiness", "instrumentalness", "tempo", "acousticness",
                  "loudness", "valence"]:
            mu, sd = self.df[c].mean(), self.df[c].std()
            if sd > 0:
                self.df[c] = (self.df[c] - mu) / sd
        return self.df[self.numerical_features]

    def get_track_id(self, artists, track_name):
        """
        Returns track_id for a given (artist, track_name) pair.
        Includes safety checks for:
          - no match (return None)
        """
        assert isinstance(artists, str) and isinstance(track_name, str)

        matches = self.df[(self.df["artists"] == artists) &
                          (self.df["track_name"] == track_name)]

        if len(matches) == 0:
            print(f"No track found for artist='{artists}', track='{track_name}'")
            return None

        return matches["track_id"].iloc[0]
