from flask import Flask, render_template, request

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
import requests
import json

app = Flask(__name__)

data  = pd.read_csv('./data/clustered_song.csv')
factory = StemmerFactory()
stemmer = factory.create_stemmer()

tfidf = TfidfVectorizer(
    min_df = 5,
    max_df = 0.95,
    max_features = 8000,
    stop_words=set(stopwords.words('indonesian'))
    )

def search_lirik(judul, penyanyi):
  judul = judul.replace(' ','%20')
  penyanyi = penyanyi.replace(' ','%20')
  response = requests.get('https://api.musixmatch.com/ws/1.1/matcher.lyrics.get?q_track='+judul+'&q_artist='+penyanyi+'&apikey=e4b17c1f018c3dbc120298833070454c')
  if (response.status_code == 200):
    jsonResponse = response.json()
    lirik = jsonResponse['message']['body']['lyrics']['lyrics_body']
  else:
    print('something wrong in request')
    lirik = ''
  return lirik

def recommend_song(array_lyric):
    predictText = tfidf.fit_transform(array_lyric)
    cosine_similarities = linear_kernel(predictText, predictText)
    cosine_similarity_scores = list(enumerate(cosine_similarities[-1]))
    cosine_similarity_scores = sorted(cosine_similarity_scores, key=lambda x: x[1], reverse=True)
    cosine_similarity_scores = cosine_similarity_scores[1:2]
    recommendation_indices = [i[0] for i in cosine_similarity_scores]
    recommendations = data['Title'].iloc[recommendation_indices]
    return recommendations

##Route##
@app.route('/')
def main():
    return render_template('tubes.html')

@app.route('/recommend', methods = ['POST'])
def recommend():
    judul = request.form['judul']
    penyanyi = request.form['penyanyi']
    lirik = search_lirik(judul, penyanyi)
    lirik_preprocessing = stemmer.stem(lirik.lower())
    array_lyric =  data.processed_lyric
    array_lyric.append(pd.Series(lirik_preprocessing,index=[400]))
    rekomendasi  = recommend_song(array_lyric)
    for name in rekomendasi:
        lagu_rekomendasi = name.replace('Lirik Lagu ','')
    get_rekomendasi_cluster =  data.loc[data['cluster'] == data['cluster'][rekomendasi.index[0]]]
    rekomendasi_lainnya = get_rekomendasi_cluster.head(20)
    data_rekomendasi = []
    for name in rekomendasi_lainnya.Title:
        data_rekomendasi.append(name.replace('Lirik Lagu ',''))
    return render_template('hasil.html',judul =''.join(judul), penyanyi=''.join(penyanyi), top_rekomendasi = lagu_rekomendasi, data_rekomendasi = data_rekomendasi )


if __name__ == "__main__":
    app.run(debug=True)
