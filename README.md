#  Email Spam Detection with Machine Learning

##  Project Overview

This project focuses on building a **Natural Language Processing (NLP) based binary classification system** to distinguish between **Spam** and **Ham (legitimate)** messages.

The project uses **text preprocessing, TF-IDF feature extraction, and machine learning classification algorithms** to identify unwanted messages.

This project was completed as part of my **Data Analytics / Machine Learning Internship Task 4**.

---

##  Objective

The main objective of this project is to develop a machine learning model that can automatically classify text messages into:

* 📨 **Ham** – Legitimate messages
*  **Spam** – Unwanted or potentially fraudulent messages

The project also evaluates different machine learning models using multiple performance metrics.

---

##  Technologies & Tools

* **Python**
* **Pandas** – Data loading and manipulation
* **NumPy** – Numerical operations
* **Matplotlib** – Data visualization
* **Seaborn** – Statistical visualization
* **NLTK** – Text preprocessing
* **Regular Expressions (re)** – Text cleaning
* **Scikit-learn** – Machine learning and evaluation
* **TF-IDF Vectorizer** – Text feature extraction
* **WordCloud** – Word frequency visualization
* **Jupyter Notebook**

---

##  Dataset

The project uses the **SMS Spam Collection Dataset**, a classic dataset containing labeled SMS messages.

Each message belongs to one of two categories:

| Label  | Description           |
| ------ | --------------------- |
| `ham`  | Legitimate message    |
| `spam` | Unwanted/spam message |

The dataset contains **5,572 messages**.

### Dataset Distribution

* **Ham:** 4,825 messages (~86.6%)
* **Spam:** 747 messages (~13.4%)

The class distribution shows that the dataset is **imbalanced**, with significantly more ham messages than spam messages.

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
Machine Learning Models
   ↓
Model Evaluation
   ↓
Confusion Matrix
   ↓
Model Comparison
```

---

##  1. Data Loading & Inspection

The dataset was loaded using Pandas.

Initial data analysis included:

* Dataset shape
* First and last records
* Data types
* Missing value detection
* Duplicate value detection
* Class distribution

Example:

```python
df = pd.read_csv("spam-task-4.csv")
```

---

##  2. Text Preprocessing

Raw text cannot be directly used by most machine learning algorithms. Therefore, the messages were cleaned and transformed before feature extraction.

The preprocessing steps include:

1. Converting text to lowercase
2. Removing punctuation and unnecessary characters
3. Splitting text into individual words
4. Removing English stopwords
5. Applying stemming using **Porter Stemmer**

### Example

```python
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()

    words = [
        ps.stem(word)
        for word in words
        if word not in stopwords.words('english')
    ]

    return " ".join(words)
```

---

##  3. TF-IDF Feature Extraction

**TF-IDF (Term Frequency–Inverse Document Frequency)** was used to convert the preprocessed text into numerical features.

TF-IDF measures the importance of a word within a document relative to the entire collection of documents.

A word receives a higher score when it is important to a particular message but does not appear frequently across the entire dataset.

```python
tfidf = TfidfVectorizer()

X = tfidf.fit_transform(df['processed'])
```

---

##  4. Train-Test Split

The dataset was divided into training and testing sets using an **80:20 split**.

* **80%** → Training data
* **20%** → Testing data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The training set is used to train the models, while the testing set evaluates their performance on unseen data.

---

##  5. Machine Learning Models

Two classification algorithms were implemented.

### 1. Multinomial Naive Bayes

Multinomial Naive Bayes is widely used for **text classification** because it works effectively with text-derived features such as word counts and TF-IDF values.

```python
nb = MultinomialNB()

nb.fit(X_train, y_train)

pred_nb = nb.predict(X_test)
```

### 2. Logistic Regression

Logistic Regression was used as an alternative binary classification model.

```python
lr = LogisticRegression()

lr.fit(X_train, y_train)

pred_lr = lr.predict(X_test)
```

---

##  6. Model Evaluation

The models were evaluated using:

* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **Confusion Matrix**

### Why Multiple Metrics?

Accuracy alone may not provide a complete picture when dealing with an **imbalanced dataset**.

Therefore, precision, recall, and F1-score were also considered to better understand the model's ability to identify spam messages.

---

##  Why is Recall Important for Spam Detection?

Recall is particularly important because it measures how many of the **actual spam messages** are correctly identified by the model.

A low spam recall means that many spam messages are being incorrectly classified as legitimate messages.

For a real-world spam detection system, improving spam recall can help prevent unwanted or potentially harmful messages from reaching users.

---

##  Confusion Matrix

A confusion matrix was used to understand the classification results in detail.

It shows:

* **True Positive (TP)** – Spam correctly identified as spam
* **True Negative (TN)** – Ham correctly identified as ham
* **False Positive (FP)** – Ham incorrectly classified as spam
* **False Negative (FN)** – Spam incorrectly classified as ham

This provides more insight into model performance than accuracy alone.

---

##  WordCloud Visualization

As an additional analysis, WordCloud visualizations can be used to identify frequently occurring words in:

* Spam messages
* Ham messages

This provides a visual understanding of the vocabulary commonly associated with each class.

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

##  Key Learnings

Through this project, I gained practical experience in:

* Natural Language Processing (NLP)
* Text preprocessing
* Stopword removal
* Stemming
* TF-IDF feature engineering
* Binary classification
* Multinomial Naive Bayes
* Logistic Regression
* Model evaluation
* Confusion matrix analysis
* Handling imbalanced classification datasets

---

##  Future Improvements

The project can be further improved by:

* Using **class balancing techniques**
* Testing **Support Vector Machine (SVM)**
* Hyperparameter tuning
* Using **n-grams** with TF-IDF
* Trying lemmatization instead of stemming
* Comparing additional NLP techniques
* Building a simple web application for real-time spam prediction

---

##  Conclusion

This project demonstrates how **Natural Language Processing and Machine Learning** can be applied to automatically classify messages as spam or legitimate.

By combining text preprocessing, TF-IDF feature extraction, and machine learning classification algorithms, the project provides a practical implementation of an NLP-based spam detection pipeline.

The project also highlights the importance of using **precision, recall, and F1-score alongside accuracy**, particularly when working with imbalanced datasets.

---

##  Author

**Prajna Mondal**

B.Tech – Computer Science & Engineering

### 🔗 Connect with Me

* LinkedIn: https://www.linkedin.com/in/prajna-mondal-1b864137a/


---

If you find this project useful, feel free to star the repository! Thank You.
