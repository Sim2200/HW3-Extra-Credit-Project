# Spotify Artist Hashing Project

## AI Usage

AI tools(Chatgpt, Claude) were used only for assistance in designing the user interface (UI) of the project. The hashing logic, algorithm implementation, and data preprocessing were developed by the members of the group.

## Project Overview
This project analyzes a Spotify music dataset and organizes songs using hashing.  
The main goal is to efficiently group songs by artist and allow quick retrieval of songs associated with a specific artist.

Hashing is implemented using Python dictionaries where the artist name acts as the key and the related song information is stored as the value.

The dataset used in this project contains more than 10,000 songs along with metadata such as track names, artist names, and artist IDs.

---

## Dataset

The dataset used in this project is too large to upload directly to GitHub.  
You can download it from Kaggle using the link below:

https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs

After downloading the dataset:

1. Extract the files.
2. Place the dataset file in the project folder before running the notebook.

Example folder structure after downloading the dataset:

```
project-folder/
│
├── Project.ipynb
├── README.md
├── hashing_report.pdf
└── dataset.csv   (downloaded from Kaggle)
```

Note: The dataset file is not included in this repository because it exceeds GitHub’s file size limits.

---

## Project Structure

```
project-folder/
│
├── Project.ipynb        # Main project notebook
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

Install the required libraries using:

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

### Step 4: Download the dataset

Download the dataset from Kaggle and place the dataset file in the project folder.

### Step 5: Open the notebook

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

## User Interface (UI)

This project includes a Streamlit-based user interface that allows users to interact with the system and search for artists and songs.

### Running the UI

To launch the interface, run the following command in the terminal:

```
streamlit run spotify_streamlit_ui.py
```

After running the command, open the link shown in the terminal (usually):

```
http://localhost:8501
```

If the UI server is already running on the network, it may also be accessible at:

```
http://192.168.1.165:8501/
```

Note: The local network address will only work for users connected to the same network.

---

## How the Program Works

1. The dataset is loaded into the program.
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
