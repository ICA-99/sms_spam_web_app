import pandas as pd

df = pd.read_csv(
    r"../data/spam.csv",
    encoding="latin-1"
)

# print(df.shape) # (5572, 5) 


# 1. Data Cleaning
# 2. EDA
# 3. Text Preprocessing
# 4. Model building
# 5. Evaluation
# 6. Improvement
# 7. Website
# 8. Deploy


# - - - - - - - - - - Data Cleaning - - - - - - - - - 

# print(df.info())
"""
 #   Column      Non-Null Count  Dtype 
---  ------      --------------  ----- 
 0   v1          5572 non-null   object
 1   v2          5572 non-null   object
 2   Unnamed: 2  50 non-null     object
 3   Unnamed: 3  12 non-null     object
 4   Unnamed: 4  6 non-null      object
"""

# Drop last three columns
df = df.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])

# print(df.shape)


# Renaming column names
df = df.rename(columns={"v1" : "target", "v2" : "text"})

# print(df.columns)


df["target"] = df["target"].map({"spam" : 1, "ham" : 0})

# print(df.head())


# check missing values

# print(df.sum()) # zero(0)


# check duplicated values

# print(df.duplicated().sum()) # 403


# remove duplicates

df = df.drop_duplicates(keep="first")

# print(df.duplicated().sum()) # 0

# print(df.shape) # (5169, 2)



# 2. - - - - - - - EDA - - - - - - - - - -

value = df["target"].value_counts()

"""
target
0    4516
1     653
"""


# Spam and Ham pie chart
import matplotlib.pyplot as plt
# plt.pie(value, labels=["ham", "spam"], autopct="%0.2f")
# plt.show() #  data is imbalanced

import nltk

df["num_characters"] = df["text"].apply(len)
# print(df.shape) # (5169, 3)
# print(df.columns) # ['target', 'text', 'num_characters']


df["num_words"] = df["text"].apply(lambda x : len(nltk.word_tokenize(x)))
# print(df.shape) # (5169, 4)
# print(df.columns) # ['target', 'text', 'num_characters', 'num_words']


df["num_sentences"] = df["text"].apply(lambda x : len(nltk.sent_tokenize(x)))
# print(df.shape) # (5169, 5)
# print(df.columns) # ['target', 'text', 'num_characters', 'num_words', 'num_sentences']



ham_describe = df[df["target"] == 0][["num_characters", "num_words", "num_sentences"]].describe()
# print(ham_describe)
"""
       num_characters    num_words  num_sentences
count     4516.000000  4516.000000    4516.000000
mean        70.459256    17.123782       1.820195
std         56.358207    13.493970       1.383657
min          2.000000     1.000000       1.000000
25%         34.000000     8.000000       1.000000
50%         52.000000    13.000000       1.000000
75%         90.000000    22.000000       2.000000
max        910.000000   220.000000      38.000000
"""


spam_describe = df[df["target"] == 1][["num_characters", "num_words", "num_sentences"]].describe()
# print(spam_describe)
"""
       num_characters   num_words  num_sentences
count      653.000000  653.000000     653.000000
mean       137.891271   27.667688       2.970904
std         30.137753    7.008418       1.488425
min         13.000000    2.000000       1.000000
25%        132.000000   25.000000       2.000000
50%        149.000000   29.000000       3.000000
75%        157.000000   32.000000       4.000000
max        224.000000   46.000000       9.000000
"""

import seaborn as sns

# ham histogram
# sns.histplot(df[df["target"] == 0]["num_characters"], color='green')


# spam histogram
# sns.histplot(df[df["target"] == 1]["num_characters"], color='red')


# plt.show()


# sns.pairplot(df, hue="target")


# x = df.drop(columns=["text"])

# sns.heatmap(x.corr(), annot=True)
# plt.show()



# - - - - - - - - - - - Data Preprocessing - - - - - - - - - - - 
"""
1. lower case
2. Tokenization
3. Remove special characters
4. Remove stopwords and punctiations
5. Stemming
"""

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


"""
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
"""
stopword_ = set(stopwords.words("english"))
lem = WordNetLemmatizer()


def transform_text(text):
    tokens = nltk.word_tokenize(text.lower())

    # Step 1: Remove stopwords & non-alphanumeric
    filtered = [word for word in tokens 
                if word.isalnum() and word not in stopword_]

    # Step 2: POS tagging on filtered words
    tagged = nltk.pos_tag(filtered)

    processed_data = []

    for word, tag in tagged:
        if tag.startswith('V'):
            pos = 'v'
        elif tag.startswith('J'):
            pos = 'a'
        elif tag.startswith('R'):
            pos = 'r'
        else:
            pos = 'n'

        processed_data.append(lem.lemmatize(word, pos))

    return " ".join(processed_data)

# print(transform_text(df["text"][10]))



# Creating new column
df["transformed_text"] = df["text"].apply(transform_text)



# Spam and Ham wordcloud
"""
from wordcloud import WordCloud

spam_wc = " ".join(df[df["target"] == 1]["transformed_text"])
ham_wc = " ".join(df[df["target"] == 0]["transformed_text"])

wc = WordCloud(
    width=1500, 
    height=1500, 
    min_font_size=15,
    background_color="white"
)

wc.generate(ham_wc)
plt.imshow(wc)
plt.axis("off")
plt.show()
"""


# Top 30 spam words barplot
"""
spam_corpus = []
top_spam_sms = df[df["target"] == 1]["transformed_text"].tolist()

for message in top_spam_sms:
    for word in message.split():
        spam_corpus.append(word)

from collections import Counter

x = pd.DataFrame(Counter(spam_corpus).most_common(30))

sns.barplot(data=x, x=x[0], y=x[1], palette="tab10", legend=False)
plt.xticks(rotation='vertical')
plt.show()

"""


"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(df["transformed_text"])

y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2, stratify=y)

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred = mnb.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(precision_score(y_test, y_pred)) 
"""