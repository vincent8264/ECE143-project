import pandas as pd
from source.recommender import Recommender

def get_user_songs():
    """
    Prompt the user to enter multiple (artist, track_name) pairs.
    User types blank lines to finish input.
    Returns a list of tuples: [(artist, track_name), ...]
    """
    print("Enter songs (artist, track_name). Leave blank to finish.\n")

    songs = []
    while True:
        artist = input("Artist: ").strip()
        if artist == "":
            break

        track = input("Track name: ").strip()
        if track == "":
            break

        songs.append((artist, track))
        print(f"Added: ({artist}, {track})\n")

    return songs


def main():
    df = pd.read_csv("data/dataset.csv")
    recommender_sys = Recommender(df)

    songs = get_user_songs()

    if not songs:
        print("No songs entered. Exiting.")
        return

    print("\nInput songs:", songs)

    # Run recommendations
    print("\nGenerating recommendations...\n")
    results = recommender_sys.recommend(songs, k=5)

    print("Top Recommendations:\n")
    print(results)


if __name__ == "__main__":
    main()
