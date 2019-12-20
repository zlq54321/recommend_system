import pickle
import tensorflow as tf
import numpy as np

load_dir = './save/model'
title_count, title_set, genres2int, features, targets_values, ratings, users, \
movies, data, movies_orig, users_orig = pickle.load(open('preprocess.p', mode='rb'))

title_col = movies['Title']
len_col = np.array([len(x) for x in title_col], dtype=np.int32)
movies['Title_size'] = len_col

# users_matrics = pickle.load(open('users_matrics.p', mode='rb'))
movie_matrics = pickle.load(open('movie_matrics.p', mode='rb'))
movieid2idx = {val[0]: i for i, val in enumerate(movies.values)}


def recommend_same_type_movie(movie_id_val, top_k=20):

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # load model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        norm_movie_matrics = tf.sqrt(tf.reduce_sum(tf.square(movie_matrics), 1, keep_dims=True))
        normalized_movie_matrics = movie_matrics / norm_movie_matrics

        # 推荐
        probs_embeddings = (movie_matrics[movieid2idx[movie_id_val]]).reshape([1, 200])
        probs_similarity = tf.matmul(probs_embeddings, tf.transpose(normalized_movie_matrics))
        sim = probs_similarity.eval()

        print("您看的电影是：{}".format(movies_orig[movieid2idx[movie_id_val]]))
        print("以下是给您的推荐：")
        p = np.squeeze(sim)
        p[np.argsort(p)[:-top_k]] = 0
        p = p / np.sum(p)
        results = set()
        while len(results) != 5:
            c = np.random.choice(3883, 1, p=p)[0]
            results.add(c)
        for val in results:
            print(val)
            print(movies_orig[val])

        return results


recommend_same_type_movie(1401, 20)





