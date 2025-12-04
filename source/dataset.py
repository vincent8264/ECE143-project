import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, df):
        assert isinstance(df, pd.DataFrame)
        self.features = ['track_id', 'artists', 'album_name', 'track_name', 'popularity', 'duration_ms', 'explicit',
                         'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                         'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'track_genre']
        assert set(self.features).issubset(df.columns)
        self.raw = df
        self.categorical_features = ['album_name', 'track_name', 'explicit', 'key', 'mode', 'time_signature']
        self.multiclass_features = ['artists', 'track_genre']
        self.numerical_features = ['popularity', 'duration_ms', 'danceability', 'energy', 'loudness', 'speechiness',
                                   'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        self.df = self.clean_data(df)

    def clean_data(self, df):
        df = df.copy()

        # there is one track where artists, album_name, and track_name are None
        df.dropna(subset=['artists', 'track_name'], inplace=True)

        # when multiple rows have the same track_id, the only difference is popularity and track_genre.
        # so use max of popularity and combine track_genre
        agg_dict = {}
        for col in self.features:
            if col == 'popularity':
                agg_dict[col] = 'max'
            elif col == 'track_genre':
                agg_dict[col] = lambda x: tuple(sorted(x.unique()))
            else:
                agg_dict[col] = 'first'
        df = df.groupby('track_id', as_index=False).agg(agg_dict)
        df = df.set_index('track_id', drop=False)

        df['artists'] = df['artists'].apply(lambda x: tuple(sorted(x.split(';'))))

        # some songs have the same artists and track_name. keep the first one for now
        df = df.drop_duplicates(subset=["artists", "track_name"], keep="first")

        for feature in self.categorical_features:
            assert df[feature].isna().sum() == 0

        for feature in self.multiclass_features:
            assert df[feature].isna().sum() == 0

        for feature in self.numerical_features:
            assert df[feature].dtype == np.int64 or df[feature].dtype == np.float64
            assert df[feature].isna().sum() == 0

        return df

    def get_string(self, track_id):
        return f"{self.df.loc[track_id]['track_name']} - {", ".join(self.df.loc[track_id, "artists"])}"

    def get_track_id(self, artists, track_name):
        """
        Returns track_id for a given (artist, track_name) pair.
        Includes safety checks for:
          - no match (return None)
        """
        assert isinstance(artists, str) and isinstance(track_name, str)

        matches = self.df[(self.df["artists"] == tuple(sorted(artists.split(';')))) &
                          (self.df["track_name"] == track_name)]

        if len(matches) == 0:
            print(f"No track found for artist='{artists}', track='{track_name}'")
            return None

        return matches["track_id"].iloc[0]
