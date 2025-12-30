import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

df = pd.read_csv('data/GrammarandProductReviews.csv')

# DATA SOURCING AND EXPLORATION


df.isnull().sum().sort_values(ascending=False)

# drop rows with missing reviews.text; can't do sentiment analysis without text
df = df.dropna(subset=['reviews.text'])

# fill missing values in brand and manufacturer with 'Unknown'
df['manufacturer'] = df['manufacturer'].fillna('Unknown')

df["reviews.title"]= df["reviews.title"].fillna('')
df["reviews.didPurchase"]= df["reviews.didPurchase"].fillna(False)
df["reviews.numHelpful"] = df["reviews.numHelpful"].fillna(0)

df.isnull().sum().sort_values(ascending=False)

df['reviews.rating'].value_counts().sort_index(ascending=False)

def label_sentiment(rating):
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"

df["sentiment"] = df["reviews.rating"].apply(label_sentiment)

# text preprocessing and lemmatization

df["full_review"] = df["reviews.title"] + " " + df["reviews.text"]

df.drop(columns=['reviews.title', 'reviews.text'], inplace=True)

df["full_review"] = (
    df["full_review"]
    .str.lower()
    .str.replace("http", "", regex=False)      # removes URL prefix
    .str.replace(r"[^a-z\s]", "", regex=True)  # keep only letters
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

import spacy
nlp = spacy.load(
    "en_core_web_sm",
    disable=["ner", "parser"]
)

df["full_review"] = [
    " ".join(
        token.lemma_
        for token in doc
        if token.is_alpha and not token.is_stop
    )
    for doc in nlp.pipe(df["full_review"], batch_size=2000)
]



# TOKENIZATION AND SEQUENCE PADDING

x_text = df["full_review"]
y = df["sentiment"]
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
y = np.array([label_map[i] for i in y], dtype=np.int32)

x_train, x_test, y_train, y_test = train_test_split(
    x_text, y, test_size=0.2, random_state=42, stratify=y)

tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>") # use top 20000 words only, replace others with <OOV> token
tokenizer.fit_on_texts(x_train) # fit tokenizer on training data

max_length = 200 # max length of sequences
x_train_seq = tokenizer.texts_to_sequences(x_train)
x_train = pad_sequences(x_train_seq, maxlen=max_length, padding='post', truncating='post')
x_test_seq = tokenizer.texts_to_sequences(x_test)
x_test = pad_sequences(x_test_seq, maxlen=max_length, padding='post', truncating='post')

# MODEL BUILDING, TRAINING AND EVALUATION

vocab_size = min(20000, len(tokenizer.word_index) + 1)

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=100, input_length=max_length, mask_zero=True),
    LSTM(128),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dense(3, activation="softmax")
])

model.compile(
    loss='sparse_categorical_crossentropy', 
    optimizer='adam',
    metrics=['accuracy']
)


early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=1,
    restore_best_weights=True
)

history = model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=5,
    batch_size=64,
    callbacks = [early_stopping]
)

# evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)


# %%
y_pred = model.predict(x_test).argmax(axis=1)
print(classification_report(y_test, y_pred))

# product-level sentiment analysis

df_test = df.iloc[-len(y_pred):].copy()

df_test["sentiment_pred"] = y_pred

df_test["sentiment_score"] = df_test["sentiment_pred"]

product_sentiment = (
    df_test.groupby(["name", "brand"], as_index=False)["sentiment_score"]
    .mean()
)


# REORDERING BASED ON RATINGS AND SENTIMENT SCORES

base_recommendations = (
    df.groupby("name")
      .agg(
          avg_rating=("reviews.rating", "mean"),
          review_count=("reviews.rating", "count")
      )
      .reset_index()
)

final_recommendations = base_recommendations.merge(
    product_sentiment,
    on="name",
    how="left"
)

final_recommendations["sentiment_score"].fillna(0, inplace=True)

final_recommendations["final_score"] = (
    0.7 * final_recommendations["avg_rating"] +
    0.3 * final_recommendations["sentiment_score"]
)

final_recommendations.sort_values(by="final_score", ascending=True)

sentiment_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

df_test["sentiment_score"] = df_test["sentiment"].map(sentiment_map)

# USER-LEVEL BRAND AND CATEGORY AFFINITY CALCULATION

brand_affinity = (
    df_test.groupby(["reviews.username", "brand"])["sentiment_score"]
      .mean()
      .reset_index()
      .rename(columns={"sentiment_score": "brand_affinity"})
)

df_test["primary_category"] = df_test["categories"].str.split(",").str[0]


category_affinity = (
    df_test
    .groupby(["reviews.username", "primary_category"])["sentiment_score"]
    .mean()
    .reset_index()
    .rename(columns={"sentiment_score": "category_affinity"})
)


product_metadata = (df[["name", "brand", "categories"]].drop_duplicates())
product_metadata["primary_category"] = (product_metadata["categories"].str.split(",").str[0])
product_metadata.drop(columns=["categories"], inplace=True)
product_metadata.head()

# FINAL RECOMMENDATION FUNCTION

def recommend_for_user(user, top_k=5):
    brand_aff = brand_affinity[brand_affinity["reviews.username"] == user]

    cat_aff = category_affinity[category_affinity["reviews.username"] == user]

    candidates = product_metadata.merge(brand_aff, on="brand", how="left") \
                             .merge(cat_aff, on="primary_category", how="left")

    candidates["brand_affinity"].fillna(0, inplace=True)
    candidates["category_affinity"].fillna(0, inplace=True)

    candidates["affinity_score"] = (
        0.6 * candidates["brand_affinity"] +
        0.4 * candidates["category_affinity"]
    )

    candidates = candidates.merge(
        final_recommendations[["name", "final_score"]],
        on="name",
        how="left"
    )

    candidates["sentiment_score"] = candidates["final_score"].fillna(0)
    candidates["final_score"] = (
        0.7 * candidates["sentiment_score"] +
        0.3 * candidates["affinity_score"]
    )

    candidates.drop(columns=["brand_affinity", "category_affinity", "reviews.username_x", "reviews.username_y"], inplace=True)
    print(candidates[candidates["affinity_score"] > 0].shape[0])
    return candidates.sort_values("final_score", ascending=False).head(top_k)


# SAVE MODEL AND ARTIFACTS 
print("\nSaving model and artifacts...")
model.save('models/sentiment_model.h5')
with open('models/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
config = {'max_length': max_length, 'label_map': label_map, 'vocab_size': vocab_size}
with open('models/config.pkl', 'wb') as f:
    pickle.dump(config, f)
final_recommendations.to_csv('models/recommendations.csv', index=False)
brand_affinity.to_csv('models/brand_affinity.csv', index=False)
category_affinity.to_csv('models/category_affinity.csv', index=False)
product_metadata.to_csv('models/product_metadata.csv', index=False)

print("Training complete! All artifacts saved.")
