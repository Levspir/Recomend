# -*- coding: utf-8 -*-
from flask import Flask
from flask import request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os



app = Flask(__name__)
port = int(os.getenv("PORT"))


data = pd.read_csv('recomend.csv', header=0, sep=',')
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, )

tfidf_matrix = tf.fit_transform(data['Описание'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


indices = pd.Series(data.index, index=data['Название']).drop_duplicates()


def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return data['Название'].iloc[movie_indices].tolist()


@app.route('/')
def hello_world():
    return 'Hello World! I am running'



@app.route('/request', methods=['GET','POST'])
def request():
    content = request.get_json()
    mess = content['title']
    return jsonify(content)



@app.route('/recomend', methods=['POST'])
def get_category():
    content = request.get_json()
    mess = content['title']
    resp = []

    recommend_answer = get_recommendations(mess)
    resp.append(
        recommend_answer
     )
    print(resp)
    return jsonify(resp)


if __name__ == '__main__':
    app.debug=True
    app.run(host='0.0.0.0', port=port)
