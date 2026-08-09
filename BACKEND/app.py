from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Load trained model and preprocessing objects
model = joblib.load('final_model.pkl')
tfidf_tags = joblib.load('tfidf_tags.pkl')
tfidf_title = joblib.load('tfidf_title.pkl')
feature_columns = joblib.load('feature_columns.pkl')


@app.route('/')
def home():
    return jsonify({"status": "API is running"})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    title = data.get('title', '')
    tags = data.get('tags', '')
    country = data.get('country', 'Unknown')
    has_known_malware = int(data.get('has_known_malware', 0))
    description = data.get(
        'description',
        'No description provided for this threat report.'
    )

    # Create numeric features
    description_length = len(description)
    tags_count = len(tags.split(',')) if tags.strip() else 0

    numeric_df = pd.DataFrame([[
        has_known_malware,
        description_length,
        tags_count
    ]], columns=[
        'has_known_malware',
        'description_length',
        'tags_count'
    ])

    # Country one-hot encoding
    country_columns = [
        col for col in feature_columns
        if col.startswith('country_')
    ]

    country_row = pd.DataFrame(
        [[0] * len(country_columns)],
        columns=country_columns
    )

    # Match the country value used during training
    country_col = f'country_{country}'

    if country_col in country_row.columns:
        country_row[country_col] = 1
    elif 'country_Other' in country_row.columns:
        country_row['country_Other'] = 1

    # Tags TF-IDF
    tags_vec = tfidf_tags.transform([tags])

    tags_df = pd.DataFrame(
        tags_vec.toarray(),
        columns=[
            f'tag_{w}'
            for w in tfidf_tags.get_feature_names_out()
        ]
    )

    # Title TF-IDF
    title_vec = tfidf_title.transform([title])

    title_df = pd.DataFrame(
        title_vec.toarray(),
        columns=[
            f'title_{w}'
            for w in tfidf_title.get_feature_names_out()
        ]
    )

    # Combine all features
    row = pd.concat([
        numeric_df,
        country_row,
        tags_df,
        title_df
    ], axis=1)

    # Ensure exact same feature order as training
    row = row.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Prediction
    prediction = model.predict(row)[0]

    probabilities = model.predict_proba(row)[0]
    classes = model.classes_

    top3_idx = np.argsort(probabilities)[-3:][::-1]

    top3 = [
        {
            "industry": classes[i],
            "confidence": round(float(probabilities[i]), 3)
        }
        for i in top3_idx
    ]

    return jsonify({
        "prediction": prediction,
        "top_3": top3
    })


if __name__ == '__main__':
    app.run(debug=True)