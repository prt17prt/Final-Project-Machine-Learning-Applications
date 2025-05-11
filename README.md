# Final-Project-Machine-Learning-Applications


<p align="center">
  <img src="images/logo_uc3m.png">
</p>



100499170: Pablo Ruiz Torralba

100496380: Jesús Navarro Benito 

100466636: Jason Nzondomyo Ngalambude


## 1. Introduction

In this project, we aim to analyze and understand the thematic structure and popularity patterns of Reddit posts related to sports. Reddit, as one of the most widely used platforms for online discussion, offers great user-generated content across numerous communities (subreddits), particularly those dedicated to sports like football, basketball, or tennis. In order to achieve this, we used some of the tools explained in class about Machine learning tools and Natural Language Processing.

## 2. Task 1: Text Preprocessing and vectorization

#### 2.1 The dataset
To build our dataset, we used the PRAW (Python Reddit API Wrapper) library to collect the top 1,000 posts from each of ten popular sports-related subreddits: sports, soccer, nba, nfl, baseball, hockey, mma, formula1, tennis, and CFB. For each post, we extracted metadata including the subreddit name, post title, number of upvotes, number of comments, post ID, and creation timestamp. Since the main focus of our project is based on the title text, we discarded other fields like the post body, which were often missing or empty.

The resulting dataset contains around 10,000 posts and was saved in CSV format for processing later.

<p align="center">
  <img src="images/csv_head.png">
</p>




We observe that from 2020 to 2022 and in 2025 are the years with most tweets, which we could take into account for the observation in the dashboard later as we have used dates for different visualizations.

#### 2.2 Preprocessing

**Language Detection**

Before preprocessing the text, we ensured all titles were in English by applying the langdetect library. Titles identified as non-English were filtered out to maintain consistency in the process and in the modeling.

**Text Cleaning and Normalization**

We applied a standard NLP preprocessing pipeline using NLTK, which included:

- Lowercasing all text.

- Removing URLs.

- Tokenizing titles into words.

- Removing stopwords and non-alphabetic tokens.

- Lemmatizing words using WordNet.

This process resulted in a new column clean_title containing the cleaned and normalized version of each title.

**Vectorization: TF-IDF Representation**

We used the TfidfVectorizer from scikit-learn to convert the cleaned titles into numerical vectors. This method assigns weights to terms based on their frequency across documents while penalizing overly common terms. We experimented with various values for min_df and max_df to find a balance between rare and frequent terms, and we included both unigrams and bigrams to capture common multi-word expressions (e.g., "world cup").

The final TF-IDF matrix was sparse, as expected, and had a shape of (n_posts, 5000), with a low density of non-zero entries, which is typical in this type of textual representation.

This vectorized data served as the foundation for subsequent modeling and clustering tasks.





## 3. Task 2: Recommender Systems

## 4. Dashboard

In this section, we have developed an interactive Dashboard using Dash. This dashboard includes several interactive components for exploring and analyzing the topics discussed in Reddit posts about sports. Here are the main features:

#### 4.1. Word Cloud for the Selected Date Range
We’ve added a date range selector that lets you choose a specific time period. The word cloud will then generate based on the titles of Reddit posts within that date range. You can also filter the word cloud by sport, allowing you to see how popular words evolve over time within the chosen sport.

#### 4.2. Topic Distribution by Highlighted Event
This section allows you to explore topics related to significant events within a sport. You can select a sport and a major event (we have slected them), and the dashboard will generate a word cloud of the most mentioned words around that event, from the date of the event to two weeks after the event. This feature is useful for analyzing how major events influence the topics discussed by users.

#### 4.3. Topic Distribution by Sport
A dropdown selector allows you to choose a sport (e.g., football, basketball) and view a bar chart showing the distribution of topics (LDA) that are most prominent in Reddit posts related to that sport. This chart helps visualize how certain topics are more common in specific sports, which is useful for understanding trends and user interests in different sports.

#### 4.4. Evolution of Popularity for Famous Athletes
We have included a graph showing the monthly evolution of mentions for famous athletes on Reddit. You can select one or more athletes (e.g., Messi, LeBron James - we added 3 important athletes for each sport), and the graph will show how their mentions have changed month by month. This chart helps visualize fluctuations in user interest over time for different athletes. 


## 5. Conclusions


## 6. References
