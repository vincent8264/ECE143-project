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
│   └── dataset.csv                 # Kaggle Spotify dataset
│
├── source/
│   ├── dataset.py
│   └── recommender.py             # Main recommender class
│   └── utils.py                   # Useful functions for displaying outputs and retrieving data
│
├── recommendation.ipynb           # Examples of running the recommendation system with different inputs
├── visualization.ipynb            # Jupyter Notebook generating figures
│
├── main.py                        # main.py for running in command line
└── README.md                      # This file
```

### **Key file: `source/recommender.py`**

Contains the `Recommender` class, including:

* feature extraction
* dataset filtering
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
from source.utils import print_songs

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

print_songs(songs)
```

### **3. Get recommendations**

```python
songs = recommender_sys.recommend(songs)
print_songs(songs)
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

* The recommender uses cosine similarity and Jaccard similarity between feature vectors to rank songs. 
* Multiple input songs are averaged into a single “preference vector.”
* Recommendations can be randomized by selecting from the top-N similar tracks. 
