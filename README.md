#  Email Spam Detection using Machine Learning & NLP

##  Overview

This project is a **Natural Language Processing (NLP) and Machine Learning-based Spam Detection system** developed to classify text messages into two categories:

*  **Ham** – Legitimate messages
*  **Spam** – Unwanted messages

The project demonstrates a complete machine learning workflow, starting from **data exploration and text preprocessing** to **TF-IDF feature extraction, model training, and performance evaluation**.

I developed this project to gain practical hands-on experience with **NLP, text classification, feature engineering, and machine learning model evaluation**.

---

##  Objective

The main objective of this project is to build a binary classification system capable of identifying whether a given message is **spam or legitimate (ham)**.

The project focuses on understanding how machine learning algorithms can process textual data and identify patterns associated with spam messages.

---

##  Tech Stack

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **NLTK**
* **Regular Expressions**
* **Matplotlib**
* **Seaborn**
* **WordCloud**
* **Jupyter Notebook / Google Colab**

---

##  Dataset

The project uses an SMS spam dataset containing **5,572 messages** with the following relevant fields:

| Column      | Description                          |
| ----------- | ------------------------------------ |
| `spamORham` | Target label indicating spam or ham  |
| `Message`   | Text message used for classification |

The dataset contains:

* **4,825 Ham messages**
* **747 Spam messages**

### Class Distribution

| Class | Count | Percentage |
| ----- | ----: | ---------: |
| Ham   | 4,825 |     86.59% |
| Spam  |   747 |     13.41% |

The distribution shows that the dataset is **imbalanced**, with legitimate messages significantly outnumbering spam messages.

---

##  Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
Data Inspection
   ↓
Class Distribution Analysis
   ↓
Text Preprocessing
   ↓
TF-IDF Feature Extraction
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Confusion Matrix
   ↓
Model Comparison
```

---

##  1. Data Loading & Exploration

I loaded the dataset using Pandas and performed an initial inspection to understand its structure.

The analysis included:

* Viewing sample records
* Checking dataset dimensions
* Checking data types
* Checking missing values
* Checking duplicate records
* Analyzing spam and ham distribution

The dataset contains **5,572 rows and 3 columns**, including an index-like column, the classification label, and the message text.

---

##  2. Text Preprocessing

Since machine learning models cannot directly work with raw text, I applied several preprocessing techniques to prepare the messages.

### Preprocessing steps:

1. Converted text to lowercase
2. Removed unnecessary characters and punctuation
3. Tokenized the messages into words
4. Removed English stopwords
5. Applied **Porter Stemming**
6. Reconstructed the processed text

This step helps reduce unnecessary noise and makes the text more suitable for feature extraction.

---

##  3. TF-IDF Feature Extraction

I used **TF-IDF (Term Frequency–Inverse Document Frequency)** to transform the processed text into numerical features.

TF-IDF assigns importance to words based on:

* How frequently a word appears in a particular message
* How frequently the word appears across the entire dataset

This allows machine learning algorithms to work with textual information in numerical form.

```python
tfidf = TfidfVectorizer()

x = tfidf.fit_transform(df['processed'])
y = df['spamORham']
```

---

##  4. Train-Test Split

The dataset was divided into training and testing sets using an **80:20 split**.

```python
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)
```

The training data was used to build the models, while the test data was used to evaluate their performance on unseen messages.

---

##  5. Machine Learning Models

I implemented and compared two machine learning algorithms.

### 1️. Multinomial Naive Bayes

Multinomial Naive Bayes is commonly used for text classification and works effectively with TF-IDF-based features.

```python
nb = MultinomialNB()

nb.fit(x_train, y_train)

pred_nb = nb.predict(x_test)
```

### 2️. Logistic Regression

Logistic Regression was used as an alternative binary classification algorithm.

```python
lr = LogisticRegression()

lr.fit(x_train, y_train)

pred_lr = lr.predict(x_test)
```

## Both models were evaluated using the same testing dataset.

##  6. Model Evaluation

I evaluated the models using multiple classification metrics:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **Confusion Matrix**

### Model Accuracy

| Model                   | Accuracy |
| ----------------------- | -------: |
| Multinomial Naive Bayes |   86.99% |
| Logistic Regression     |   86.99% |

Both models achieved approximately **86.99% accuracy** in the current implementation.

### Important Observation

Although the overall accuracy is around 87%, the dataset is imbalanced and the current models show a **low recall for the spam class**.

This demonstrates an important machine learning lesson:

> **Accuracy alone is not always sufficient, especially for imbalanced classification problems.**

For spam detection, correctly identifying actual spam messages is particularly important, so recall should be carefully considered alongside accuracy.

---

##  Why is Recall Important?

Recall measures how many of the actual spam messages are correctly identified by the model.

A low spam recall means that some spam messages are incorrectly classified as legitimate messages.

Therefore, for a practical spam detection system, improving **spam recall** would be an important area for further development.

---

##  7. Confusion Matrix

I also used a confusion matrix to understand the classification results in more detail.

The confusion matrix helps identify:

* True Positives
* True Negatives
* False Positives
* False Negatives

This provides a clearer picture of how well the model distinguishes between spam and ham messages.

---

##  8. WordCloud

As an additional visualization, WordCloud can be used to explore frequently occurring words in spam and ham messages.

This provides a simple visual representation of the vocabulary commonly associated with each category.

---

##  Project Explanation Video

I also created a **project explanation video** where I walk through the complete implementation and explain the major steps of the project.

The video covers:

* Project objective
* Dataset and class distribution
* Text preprocessing
* TF-IDF feature extraction
* Machine learning models
* Model evaluation
* Confusion matrix
* Key observations and learnings

###  Video Links

* **GitHub:** This repository contains the complete notebook and project files.
* **LinkedIn:** [Add your LinkedIn post/video link]
* **Project Walkthrough Video:** [Add your video link]

---

##  Key Learnings

Through this project, I gained practical experience in:

* Natural Language Processing
* Text preprocessing
* Stopword removal
* Stemming
* TF-IDF feature engineering
* Binary classification
* Multinomial Naive Bayes
* Logistic Regression
* Model evaluation
* Confusion matrix interpretation
* Working with imbalanced datasets

---

##  Future Improvements

Some possible improvements for this project include:

* Improving spam recall
* Applying class-balancing techniques
* Using **Support Vector Machine (SVM)**
* Hyperparameter tuning
* Experimenting with TF-IDF n-grams
* Comparing stemming with lemmatization
* Testing advanced NLP techniques
* Deploying the model as a simple web application

---

##  Project Structure

```text
Email-Spam-Detection/
│
├── OASIS_task_4.ipynb
├── spam-task-4.csv
└── README.md
```

---

##  Conclusion

This project demonstrates a complete **NLP-based machine learning pipeline for spam detection**.

Starting with raw text data, I performed data exploration and preprocessing, converted the messages into numerical features using TF-IDF, trained two classification models, and evaluated their performance using multiple metrics.

An important takeaway from this project is that **model performance should not be judged by accuracy alone**. For an imbalanced spam classification problem, metrics such as recall, precision, and F1-score provide valuable additional insights.

Overall, this project strengthened my practical understanding of **Python, NLP, feature engineering, machine learning classification, and model evaluation**.

---

##  Author

### Prajna Mondal

**B.Tech – Computer Science & Engineering**

Interested in **Data Analytics, Machine Learning, Python, SQL, and Data Science**.


 **LinkedIn:** [Add your LinkedIn profile link]

---

 If you found this project interesting, feel free to explore the repository and share your feedback! Thank You.
