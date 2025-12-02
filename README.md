# ECE 143 Song Recommendation
## Jiaguan Pan, Justin Nguyen, Tien-Ning Lee, Vincent Kao, Yu-Shu Chen

This project implements a simple content-based music recommender system using Spotify-style audio features. 
Given one or more `(artist, track_name)` pairs, the system computes similarity scores between tracks and returns the top recommended songs.

---

## 📁 Project Structure

```
project_root/
│
├── data/
│   └── dataset.csv                 # Spotify dataset
│
├── source/
│   ├── __init__.py
│   └── recommender.py             # Main recommender class
│   └── similarity.py              # Similarity functions for the recommender
│
├── notebooks/
│   └── experiments.ipynb          # Optional: your experimentation
│
├── main.py                        # main.py for running in command line
└── README.md                      # This file
```

### **Key file: `source/recommender.py`**

Contains the `Recommender` class, including:

* feature extraction
* similarity computation
* input validation
* multi-song recommendation
* random-sampling from the top-N matches

---

## ▶️ How to Run

### **1. Load your dataset**

```python
import pandas as pd
from source.recommender import Recommender

df = pd.read_csv("data/tracks.csv")
recommender_sys = Recommender(df)
```

### **2. Provide input songs**

```python
songs = [
    ("Justin Bieber", "Sorry"),
    ("Taylor Swift", "Wildest Dreams"),
    ("Eminem", "The Real Slim Shady"),
    ("Dr. Dre;Snoop Dogg", "Still D.R.E.")
]

print(songs)
```

### **3. Get recommendations**

```python
songs = recommender_sys.recommend(songs)
print(songs)
```

This returns a list of the recommended tracks.

---

## 📦 Third-Party Modules Used

This project uses the following external Python packages:

| Module     | Purpose                                  |
| ---------- | ---------------------------------------- |
| **pandas** | DataFrame handling and lookup            |
| **numpy**  | Numerical computation, matrix operations |

Make sure they are installed:

```bash
pip install numpy pandas
```

---

## 📝 Notes

* The recommender uses cosine similarity between flattened feature vectors. 
* Optional: artist matching provides a small bonus in similarity scoring. 
* Multiple input songs are averaged into a single “preference vector.”
* Recommendations can be randomized by selecting from the top-N similar tracks. 
