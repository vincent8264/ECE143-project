from source.dataset import Dataset


def print_songs(songs):
    assert isinstance(songs, list) and all(len(song) == 2 for song in songs)
    assert all((isinstance(artists, str) or isinstance(artists, list)) and isinstance(track_name, str)
               for artists, track_name in songs)

    for artists, track_name in songs:
        if isinstance(artists, str):
            print(f"{track_name} - {artists}")
        else:
            print(f"{track_name} - {", ".join(artists)}")


def get_unique_artists(dataset):
    assert isinstance(dataset, Dataset)
    return set(artist for tup in dataset.df["artists"] for artist in tup)


def get_unique_genres(dataset):
    assert isinstance(dataset, Dataset)
    return set(artist for tup in dataset.df["track_genre"] for genre in tup)
