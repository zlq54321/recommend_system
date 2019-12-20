import tensorflow as tf
import pickle

title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig \
    = pickle.load(open('preprocess.p', mode='rb'))

embed_dim = 32
HIDDEN_SIZE = embed_dim
uid_max = max(features.take(0, 1)) + 1
gender_max = max(features.take(2, 1)) + 1
age_max = max(features.take(3, 1)) + 1
job_max = max(features.take(4, 1)) + 1
movie_id_max = max(features.take(1, 1)) + 1
movie_category_max = max(genres2int.values()) + 1
movies_titles_max = len(title_set)

combiner = 'sum'
sentences_size = title_count
window_sizes = {2, 3, 4, 5}
filter_num = 8
movieid2idx = {val[0]: i for i, val in enumerate(movies.values)}


def save_params(params):
    pickle.dump(params, open('./params.p', 'wb'))


def load_params():
    return pickle.load(open('./params.p', mode='rb'))


def get_inputs():
    uid = tf.placeholder(tf.int32, [None, 1], name='uid')
    user_gender = tf.placeholder(tf.int32, [None, 1], name='user_gender')
    user_age = tf.placeholder(tf.int32, [None, 1], name='user_age')
    user_job = tf.placeholder(tf.int32, [None, 1], name='user_job')

    movie_id = tf.placeholder(tf.int32, [None, 1], name='movie_id')
    movie_categories = tf.placeholder(tf.int32, [None, 18], name='movie_categories')
    movie_titles = tf.placeholder(tf.int32, [None, sentences_size], name='movie_titles')
    title_size = tf.placeholder(tf.int32, [None, 1], name='title_size')
    targets = tf.placeholder(tf.int32, [None, 1], name='targets')
    LearningRate = tf.placeholder(tf.float32, name='LearningRate')
    dropout_keep_pob = tf.placeholder(tf.float32, name='dropout_keep_prob')

    return uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, title_size, \
            targets, LearningRate, dropout_keep_pob


def get_user_embedding(uid, user_gender, user_age, user_job):
    with tf.name_scope('user_embedding'):
        uid_embed_matrix = tf.Variable(tf.random_uniform([uid_max, embed_dim], -1, 1),
                                       name='uid_embed_matrix')
        uid_embed_layer = tf.nn.embedding_lookup(uid_embed_matrix, uid,
                                                 name='uid_embed_layer')

        gender_embed_matrix = tf.Variable(tf.random_uniform([gender_max, embed_dim // 2], -1, 1),
                                          name='gender_embed_matrix')
        gender_embed_layer = tf.nn.embedding_lookup(gender_embed_matrix, user_gender,
                                                    name='gender_embed_layer')

        age_embed_matrix = tf.Variable(tf.random_uniform([age_max, embed_dim // 2], -1, 1),
                                       name='age_embed_matrix')
        age_embed_layer = tf.nn.embedding_lookup(age_embed_matrix, user_age,
                                                 name='age_embed_layer')

        job_embed_matrix = tf.Variable(tf.random_uniform([job_max, embed_dim // 2], -1, 1),
                                       name='job_embed_matrix')
        job_embed_layer = tf.nn.embedding_lookup(job_embed_matrix, user_job,
                                                 name='job_embed_layer')

    return uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer


def get_user_feature_layer(uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer):
    with tf.name_scope('user_fc'):
        # 第一层
        uid_fc_layer = tf.layers.dense(uid_embed_layer, embed_dim, name='uid_fc_layer', activation=tf.nn.relu)
        gender_fc_layer = tf.layers.dense(gender_embed_layer, embed_dim, name='gender_fc_layer', activation=tf.nn.relu)
        age_fc_layer = tf.layers.dense(age_embed_layer, embed_dim, name='age_fc_layer', activation=tf.nn.relu)
        job_fc_layer = tf.layers.dense(job_embed_layer, embed_dim, name='job_fc_layer', activation=tf.nn.relu)
        user_combine_layer = tf.concat([uid_fc_layer, gender_fc_layer, age_fc_layer, job_fc_layer], 2)
        # 第二层
        user_combine_layer = tf.contrib.layers.fully_connected(user_combine_layer, 200, tf.tanh)
        user_combine_layer_flat = tf.reshape(user_combine_layer, [-1, 200])

    return user_combine_layer, user_combine_layer_flat


def get_movie_id_embed_layer(movie_id):
    with tf.name_scope("movie_embedding"):
        movie_id_embed_matrix = tf.Variable(tf.random_uniform([movie_id_max, embed_dim], -1, 1),
                                            name="movie_id_embed_matrix")
        movie_id_embed_layer = tf.nn.embedding_lookup(movie_id_embed_matrix, movie_id,
                                                      name="movie_id_embed_layer")
    return movie_id_embed_layer


def get_movie_categories_layers(movie_categories):
    with tf.name_scope('movie_categories_layers'):
        movie_categories_embed_matrix = tf.Variable(tf.random_uniform([movie_category_max, embed_dim], -1, 1),
                                                    name='movie_categories_embed_matrix')
        movie_category_embed_layer = tf.nn.embedding_lookup(movie_categories_embed_matrix,
                                                            movie_categories,
                                                            name='movie_categories_embed_layer')
        if combiner == 'sum':
            movie_category_embed_layer = tf.reduce_sum(movie_category_embed_layer, axis=1, keep_dims=True)
        elif combiner == 'mean':
            movie_category_embed_layer = tf.reduce_mean(movie_category_embed_layer, axis=1, keep_dims=True)

        return movie_category_embed_layer


def get_movie_rnn_layer(movie_titles, movie_title_size, dropout_keep_prob):
    with tf.name_scope('movie_embedding'):
        movie_title_embed_matrix = tf.Variable(tf.random_uniform([movies_titles_max, embed_dim], -1, 1),
                                               name='movie_title_embed_matrix')
        movie_title_embed_layer = tf.nn.embedding_lookup(movie_title_embed_matrix,
                                                         movie_titles,
                                                         name='movie_title_embed_layer')
        # movie_title_embed_layer_expand = tf.expand_dims(movie_title_embed_layer, -1)

        # 只有训练时使用dropout
        dropout_emb = tf.nn.dropout(movie_title_embed_layer, dropout_keep_prob)

    with tf.name_scope('rnn'):
        basic_cell = tf.nn.rnn_cell.BasicLSTMCell
        cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(basic_cell(HIDDEN_SIZE),
                                                                          output_keep_prob=dropout_keep_prob)
                                            for _ in range(2)])
        movie_title_size = tf.reshape(movie_title_size, [-1])
        outputs, enc_state = tf.nn.dynamic_rnn(cell, dropout_emb, movie_title_size, dtype=tf.float32)
        output = outputs[:, -1, :]
        output = tf.reshape(output, [-1, 1, HIDDEN_SIZE])

        fc_output = tf.layers.dense(output, HIDDEN_SIZE, name="movie_title_fc_layer",
                                    activation=tf.nn.relu)

        return fc_output


def get_movie_cnn_layer(movie_titles, dropout_keep_prob):
    with tf.name_scope('movie_embedding'):
        movie_title_embed_matrix = tf.Variable(tf.random_uniform([movies_titles_max, embed_dim], -1, 1),
                                               name='movie_title_embed_matrix')
        movie_title_embed_layer = tf.nn.embedding_lookup(movie_title_embed_matrix,
                                                         movie_titles,
                                                         name='movie_title_embed_layer')
        movie_title_embed_layer_expand = tf.expand_dims(movie_title_embed_layer, -1)

        # 不同的卷积核做卷积核最大池化
        pool_layer_1st = []
        for window_size in window_sizes:
            with tf.name_scope('movie_txt_conv_maxpool_{}'.format(window_size)):
                filter_weights = tf.Variable(tf.truncated_normal([window_size, embed_dim, 1, filter_num], stddev=0.1),
                                             name='filter_weights')
                filter_bias = tf.Variable(tf.constant(0.1, shape=[filter_num]), name='filter_bias')

                conv_layer = tf.nn.conv2d(movie_title_embed_layer_expand,
                                          filter_weights,
                                          strides=[1, 1, 1, 1],
                                          padding='VALID',
                                          name='conv_layer')
                relu_layer = tf.nn.relu(tf.nn.bias_add(conv_layer, filter_bias), name='relu_layer')
                maxpool_layer = tf.nn.max_pool(relu_layer,
                                               ksize=[1, sentences_size - window_size + 1, 1, 1],
                                               strides=[1, 1, 1, 1],
                                               padding='VALID',
                                               name='maxpool_layer')
                pool_layer_1st.append(maxpool_layer)

        # dropout层
        with tf.name_scope('pool_dropout'):
            pool_layer = tf.concat(pool_layer_1st, axis=3, name='pool_layer')
            max_num = len(window_sizes) * filter_num
            pool_layer_flat = tf.reshape(pool_layer, [-1, 1, max_num], name='pool_layer_flat')
            dropout_layer = tf.nn.dropout(pool_layer_flat, dropout_keep_prob, name='dropout_layer')

        return pool_layer_flat, dropout_layer


def get_movie_feature_layer(movie_id_embed_layer, movie_categories_embed_layer, dropout_layer):
    with tf.name_scope("movie_fc"):
        # 第一层全连接
        movie_id_fc_layer = tf.layers.dense(movie_id_embed_layer, embed_dim, name="movie_id_fc_layer",
                                            activation=tf.nn.relu)
        movie_categories_fc_layer = tf.layers.dense(movie_categories_embed_layer, embed_dim,
                                                    name="movie_categories_fc_layer", activation=tf.nn.relu)
        movie_combine_layer = tf.concat([movie_id_fc_layer, movie_categories_fc_layer, dropout_layer], 2)  # (?, 1, 96)

        # 第二层全连接
        movie_combine_layer = tf.contrib.layers.fully_connected(movie_combine_layer, 200, tf.tanh)  # (?, 1, 200)

        movie_combine_layer_flat = tf.reshape(movie_combine_layer, [-1, 200])
    return movie_combine_layer, movie_combine_layer_flat


def main(argv=None):
    print('^o^')


if __name__ == '__main__':
    main()






