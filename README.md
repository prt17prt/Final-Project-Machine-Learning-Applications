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

We can also check the distribution per sport, so each sport contains around 1000 posts as mentioned before.
<p align="center">
  <img src="images/posts_p_subreddit.png">
</p>

Finally, we will plot a graph to see the posts per month in our csv file.

<p align="center">
  <img src="images/posts_p_month.png">
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



#### 2.3 Text Representation and Regression Evaluation

We experimented with four different ways of turning post titles into numerical vectors: TF-IDF, Word2Vec, Doc2Vec, and LDA. These representations were then evaluated by how well they helped a Ridge regression model predict the popularity of a post (measured as log1p(upvotes)) using 5-fold cross-validation.

**Vectorization: TF-IDF Representation**

We used the TfidfVectorizer from scikit-learn to convert the cleaned titles into numerical vectors. This method assigns weights to terms based on their frequency across documents while penalizing overly common terms.We used a TfidfVectorizer with unigrams and bigrams, limiting the vocabulary to words that appear in at least 5 posts but not in more than 70% of them. This resulted in 4,144 features across 9,802 posts.

The matrix is very sparse (~0.20% non-zero values), which is expected for TF-IDF. Analyzing the idf values, the most common words were general sports terms like "game", "team", or "player", while rarer ones included "achilles" or "acquire".

The TF-IDF representation achieved a cross-validated RMSE of 0.9157, which served as our baseline.


**Word2Vec**

We trained a Word2Vec model locally using Gensim with a 400-dimensional embedding space. Each title was represented by averaging the vectors of its words.

Some documents ended up with zero vectors (about 0.6%) because all their words were out-of-vocabulary (too rare). The performance was slightly worse than TF-IDF, with an RMSE of 0.9491 ± 0.4159.


**Doc2Vec**

Unlike Word2Vec, which averages word vectors, Doc2Vec directly learns document-level embeddings. We used the Distributed Memory (DM) variant with 400 dimensions.

This time, no documents had zero vectors, and the representation performed slightly better than the others, reaching an RMSE of 0.8970 ± 0.3992.


**LDA (Latent Dirichlet Allocation)**

For topic modeling, we trained an LDA model using Gensim. After testing several values for the number of topics, we selected k = 20 based on coherence scores. Each title was then represented by a 20-dimensional topic distribution vector.

We also analyzed topic distributions across subreddits to spot differences in thematic focus.

In regression, the LDA representation yielded an RMSE of 0.8626 ± 0.3944, slightly better than Doc2Vec and Word2Vec.

At first we thought that there were going to be 10 topics as we had 10 subreddits but after removing stopwrods and noise, as many terms in sports can be confused for other sports we ended up with other results relying in the coherence matrix. 

<p align="center">
  <img src="images/lda_graph.png">
</p>

We can observe that most of the topics have the same importance for most of the sports. However, we can highlight some of them, like topic 15 for CFB, topic 19 for mma or topic 2 for nfl.


## 3. Task 2:  Machine Learning model

### Task 2.3 – Recommender Systems
In this task, we built and compared three recommendation approaches:

Content‐based k-NN using dense Sentence-Transformer embeddings.

Item-based Collaborative Filtering (CF) via Surprise’s KNNBasic.

Latent-factor CF with Implicit’s Alternating Least Squares (ALS) and Surprise’s SVD.

Our goal was to measure predictive accuracy (RMSE/MAE) on held-out ratings and ranking quality (Precision@5). 

### 1. Content-Based k-NN

Representation: We encoded each post’s cleaned title into a 384-dimensional SentenceTransformer embedding (all-MiniLM-L6-v2).

Method: A simple k-NN index (cosine distance) returns the top-5 most similar posts for any query, filtering out near-duplicates (similarity ≥ 0.9).

Example Output: ![image](https://github.com/user-attachments/assets/41c7f7bd-cc5b-4d3f-9a0d-2f1eddf44dd6)

Evaluation (Precision@5): 0.0031
On our held-out “like” test set (binary comment/upvote interactions), only ~0.3 % of true positives appeared in the top-5 recommendations. This low hit rate underscores the limitations of title text alone for recalling user preferences.


### 2. Item-Based KNN (Collaborative)
We constructed a binary user–post interaction matrix from comment upvotes (rating=1) and trained Surprise’s KNNBasic with cosine similarity (item-based mode) 
We evaluated RMSE and MAE via 5-fold CV on the positive interactions:

Algorithm	RMSE	MAE
Item-KNN	0.4983	0.4961

Despite zero MAE on folds where no variation existed, the RMSE of ~0.498 indicates modest predictive accuracy in reconstructing held-out “likes.”

### 3. Latent-Factor CF

### 3.1 ALS (Implicit)

We built an Alternating Least Squares model using the Implicit library, factoring our sparse confidence matrix C = 1 + 𝛼.R with α=10, 50 factors, regularization 0.01, and 20 iterations.

ALS scores with fixed hyperparameters: 

<p align="center"> <img src="images/als_scores.jpeg" alt="Fixed-hyperparam ALS scores"> </p>
Hyperparameter grid-search over factors 
{
20
,
50
,
100
}
{20,50,100}, regularization 
{
0.001
,
0.01
,
0.1
}
{0.001,0.01,0.1}, iterations 
{
10
,
20
,
50
}
{10,20,50}, 
𝛼
∈
{
1
,
5
,
10
}
α∈{1,5,10} yielded no gain in Precision@5, but modest RMSE improvements.

ALS scores with hyperparameter tunning: 

<p align="center"> <img src="images/als_scores_hyperparam.jpeg" alt="ALS hyperparam search results"> </p>

As we can see it performs better with fixed ALS parameters than using a grid of hyperparameters and searching for the best combination

### 3.2 SVD (Explicit)
Using Surprise’s SVD on the same binary matrix, we now measured:

Algorithm	RMSE	MAE
SVD	0.0282	0.0113

This explicit-feedback matrix factorization substantially outperforms both item-KNN and ALS in rating reconstruction.

### 4. Comparative Summary
Method	RMSE	MAE	Precision@5
Content-Based k-NN	—	—	0.0031
Item-based KNN	0.4983	0.4961	0.0040
Implicit ALS	~0.3954–0.3983	~0.3590–0.3643	0.0000
SVD (Surprise)	0.0282	0.0113	—

SVD yields the lowest RMSE/MAE, indicating that explicit-feedback matrix factorization best reconstructs user preferences here.

Content-based and implicit ALS both show near-zero Precision@5, highlighting the challenge of predicting single “likes” using only titles or purely implicit signals.

Item-KNN offers only a marginal precision gain, reflecting the sparsity of our interaction data.

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

## 5. Problems encountered 

### 1. Package Incompatibilities Preventing In-Notebook Visualization
During development, we discovered that key visualization libraries (notably PyLDAvis) could not be installed or loaded together with our primary computation stack due to conflicting dependencies (e.g. scipy vs. numpy versions). As a result, we were unable to render the interactive LDA topic map directly within our main notebook environment.

Workaround: We spun up a separate Conda environment with compatible package versions just to run the PyLDAvis code, captured a screenshot of the resulting topic‐model dashboard.

![image](https://github.com/user-attachments/assets/55716b50-36b3-4edf-aaa4-85a552f9ba96)


### 2. Limited Interaction Data Harming ALS Training
Because extracting user–post interactions from Reddit (via PRAW and Pushshift) proved extremely time- and rate-limited, we restricted ourselves to merely 50 top posts per subreddit and a cap on comment depth. This resulted in a very sparse user×item confidence matrix (only ~8,400 positive interactions for ~500 posts), which prevented our ALS model from learning robust latent factors. Consequently, ALS achieved near-zero Precision@5 on held-out “likes,” and its RMSE improvements were marginal.

Also due to the poor performance, we sometimes deleted code that maybe was not wrong but we thought it was due to poorly results, we later found out that the problem was not on the code but on poor data primarily, that is why in the code itself there won´t be some results mentioned in the report, but we thought it would be good to include those results as it was part of the work in the project 

Lesson Learned: With a larger, more balanced interaction dataset (e.g. scraping >1,000 posts per subreddit), we expect ALS (and other CF methods) to converge to significantly better ranking and prediction performance.

## 6. Conclusions

1. Effective Text Representations Enhance Predictive Power
Among the four vectorization methods explored—TF-IDF, Word2Vec, Doc2Vec, and LDA—the topic-based LDA embedding consistently delivered the lowest RMSE in our regression experiments. This demonstrates that capturing higher-level thematic structure can provide stronger signals for predicting post popularity than purely lexical features or averaged word embeddings.

2. Data Sparsity Limits Collaborative Models
Our collaborative filtering pipelines (item-based KNN, ALS, and SVD) highlighted the challenges of learning from sparse interaction matrices. With only ~8,400 comment-derived “likes” over 500 posts, both ALS and item-KNN achieved near-zero Precision@5, and ALS’s RMSE improvements were marginal. In contrast, explicit SVD on the same binary matrix yielded strong rating reconstruction (RMSE ≈ 0.028), suggesting that richer or more plentiful feedback (e.g., upvotes, multi-level ratings) is critical for robust CF.

3. Transformer Embeddings Offer Flexible Content-Based Recommendations
Leveraging SentenceTransformer embeddings and a simple k-NN index allowed us to build a content-based recommender with minimal domain-specific tuning. Although Precision@5 was low (≈ 0.003), this approach remains appealing for its ease of integration and interpretability, especially when metadata or user profiles are unavailable.

4. Hyperparameter Tuning Yields Diminishing Returns under Data Constraints
Grid searches over TF-IDF thresholds, embedding dimensions, and ALS factors/regularization revealed that, beyond a certain point, further tuning produces only incremental gains when training data is limited. This underlines the importance of scaling up data collection early in the project timeline.

5. Interactive Dashboards Foster Insightful Exploration
The Dash-based dashboard—combining LDA topic distributions, word clouds, and athlete-mention trends—transforms raw model outputs into actionable visuals. Even static captures of the PyLDAvis map provided valuable thematic insights that enrich the overall analytical narrative.

## 7. References
