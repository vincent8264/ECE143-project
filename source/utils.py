def print_songs(songs):
    for artists, track_name in songs:
        if isinstance(artists, str):
            print(f"{track_name} - {artists}")
        else:
            print(f"{track_name} - {", ".join(artists)}")