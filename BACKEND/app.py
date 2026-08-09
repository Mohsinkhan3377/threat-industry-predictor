from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

model = joblib.load('final_model.pkl')
scaler = joblib.load('scaler.pkl')
tfidf_tags = joblib.load('tfidf_tags.pkl')
tfidf_title = joblib.load('tfidf_title.pkl')
top_countries = joblib.load('top_countries.pkl')
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
    description = data.get('description', 'No description provided for this threat report.')

    description_length = len(description)
    tags_count = len(tags.split(',')) if tags.strip() else 0

    numeric_df = pd.DataFrame([[description_length, tags_count]],
                               columns=['description_length', 'tags_count'])
    numeric_scaled = scaler.transform(numeric_df)

    country_final = country if country in top_countries else 'Other'
    country_cols = [c for c in feature_columns if c.startswith('country_')]
    country_row = pd.DataFrame([[0]*len(country_cols)], columns=country_cols)
    col_name = f'country_{country_final}'
    if col_name in country_row.columns:
        country_row[col_name] = 1

    tags_vec = tfidf_tags.transform([tags])
    tags_df = pd.DataFrame(tags_vec.toarray(),
                            columns=[f'tag_{w}' for w in tfidf_tags.get_feature_names_out()])

    title_vec = tfidf_title.transform([title])
    title_df = pd.DataFrame(title_vec.toarray(),
                             columns=[f'title_{w}' for w in tfidf_title.get_feature_names_out()])

    row = pd.DataFrame([[has_known_malware]], columns=['has_known_malware'])
    row['description_length'] = numeric_scaled[0][0]
    row['tags_count'] = numeric_scaled[0][1]
    row = pd.concat([row, country_row, tags_df, title_df], axis=1)

    row = row.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0]
    classes = model.classes_
    top3_idx = np.argsort(probabilities)[-3:][::-1]
    top3 = [{"industry": classes[i], "confidence": round(float(probabilities[i]), 3)} for i in top3_idx]

    return jsonify({
        "prediction": prediction,
        "top_3": top3
    })

if __name__ == '__main__':
    app.run(debug=True)