# Spotify Artist Hashing Project

## AI Usage

AI tools(Chatgpt, Claude) were used only for assistance in designing the user interface (UI) of the project. The hashing logic, algorithm implementation, and data preprocessing were developed by the members of the group.

---

## Project Overview
This project analyzes a Spotify music dataset and organizes songs using hashing.  
The main goal is to efficiently group songs by artist and allow quick retrieval of songs associated with a specific artist.

Hashing is implemented using Python dictionaries where the artist name acts as the key and the related song information is stored as the value.

The dataset used in this project contains more than 10,000 songs along with metadata such as track names, artist names, and artist IDs.

---

## Project Structure

```
project-folder/
│
├── Project.ipynb        # Main project notebook
├── dataset.csv          # Spotify dataset
├── README.md            # Instructions for running the project
├── hashing_report.pdf   # Explanation of hashing implementation
└── ui_code.py           # User interface code
```

---

## Requirements

Before running the project, make sure the following are installed:

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib

You can install the required libraries using:

```
pip install pandas numpy scikit-learn matplotlib
```

---

## How to Run the Project

### Step 1: Clone the repository

```
git clone <repository-link>
```

### Step 2: Navigate to the project folder

```
cd project-folder
```

### Step 3: Install required libraries

```
pip install pandas numpy scikit-learn matplotlib
```

### Step 4: Open the notebook

Run the following command:

```
jupyter notebook
```

Then open the file:

```
Project.ipynb
```

Run all the cells sequentially to execute the project.

---

## How the Program Works

1. The dataset is first loaded into the program.
2. Artist names are cleaned and normalized to avoid duplicates caused by formatting differences.
3. A hash table (Python dictionary) is created where the artist name is used as the key.
4. Each artist key stores songs associated with that artist.
5. Songs are categorized into:
   - Solo songs
   - Main artist songs
   - Featured songs
6. This structure allows fast lookup of songs by artist without scanning the entire dataset.

---

## Authors

- Simran Kharbanda (UID: 122283671)
- Shashank Ashoka (UID: 122241329)
