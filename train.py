from useful_fun import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import datetime
import numpy as np
import os

# Number of Epochs
num_epochs = 1
batch_size = 256
dropout_keep = 0.5
learning_rate = 0.0001
show_every_n_batches = 20
save_dir = './save/model'

tf.reset_default_graph()
train_graph = tf.Graph()
with train_graph.as_default():
    # 获取输入占位符
    uid, user_gender, user_age, user_job, \
        movie_id, movie_categories, movie_titles, title_size, \
        targets, lr, dropout_keep_prob = get_inputs()
    # 获取User的4个嵌入向量
    uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer = get_user_embedding(uid,
                                                                                               user_gender,
                                                                                               user_age,
                                                                                               user_job)
    # 得到用户特征
    user_combine_layer, user_combine_layer_flat = get_user_feature_layer(uid_embed_layer,
                                                                         gender_embed_layer,
                                                                         age_embed_layer,
                                                                         job_embed_layer)
    # 获取电影ID的嵌入向量
    movie_id_embed_layer = get_movie_id_embed_layer(movie_id)
    # 获取电影类型的嵌入向量
    movie_categories_embed_layer = get_movie_categories_layers(movie_categories)
    # 获取电影名的特征向量
    rnn_output = get_movie_rnn_layer(movie_titles, title_size, dropout_keep_prob)
    # pool_layer_flat, dropout_layer = get_movie_cnn_layer(movie_titles, dropout_keep_prob)
    # 得到电影特征
    movie_combine_layer, movie_combine_layer_flat = get_movie_feature_layer(movie_id_embed_layer,
                                                                            movie_categories_embed_layer,
                                                                            rnn_output)
    # movie_combine_layer, movie_combine_layer_flat = get_movie_feature_layer(movie_id_embed_layer,
    #                                                                        movie_categories_embed_layer,
    #                                                                        dropout_layer)

    # 协同滤波部分
    with tf.name_scope('inference'):
        inference = tf.reduce_sum(user_combine_layer_flat * movie_combine_layer_flat, axis=1)
        inference = tf.expand_dims(inference, axis=1)

    with tf.name_scope('loss'):
        loss = tf.losses.mean_squared_error(targets, inference)
        # loss = tf.reduce_mean(lost)

    global_step = tf.Variable(0, name="global_step", trainable=False)
    # train_op = tf.train.AdamOptimizer(lr).minimize(cost, global_step=global_step)
    optimizer = tf.train.AdamOptimizer(lr)
    gradients = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(gradients, global_step=global_step)


def get_batch(Xs, ys, batch_size):
    for start in range(0, len(Xs)):
        end = min(start + batch_size, len(Xs))
        yield Xs[start: end], ys[start: end]


losses = {'train': [], 'test': []}

with tf.Session(graph=train_graph) as sess:
    """
    grad_summaries = []
    for g, v in gradients:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name.replace(':', '_')), g)
            sparsity_summary = tf.summary.scalar('{}/grad/sparsity'.format(v.name.replace(':', '_')),
                                                 tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)

    grad_summaries_merged = tf.summary.merge(grad_summaries)

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", loss)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Inference summaries
    inference_summary_op = tf.summary.merge([loss_summary])
    inference_summary_dir = os.path.join(out_dir, "summaries", "inference")
    inference_summary_writer = tf.summary.FileWriter(inference_summary_dir, sess.graph)
    """

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for epoch_i in range(num_epochs):
        train_X, test_X, train_y, test_y = train_test_split(features,
                                                            targets_values,
                                                            test_size=0.2,
                                                            random_state=0)
        train_batches = get_batch(train_X, train_y, batch_size)
        test_batches = get_batch(test_X, test_y, batch_size)

        for batch_i in range(len(train_X) // batch_size):
            x, y = next(train_batches)

            categories = np.zeros([batch_size, 18])
            titles = np.zeros([batch_size, sentences_size])
            title_sizes = np.zeros([batch_size, 1])

            for i in range(batch_size):
                categories[i] = x.take(6, 1)[i]
                titles[i] = x.take(5, 1)[i]
                title_sizes[i] = x.take(7, 1)[i]

            feed = {
                uid: np.reshape(x.take(0, 1), [batch_size, 1]),
                user_gender: np.reshape(x.take(2, 1), [batch_size, 1]),
                user_age: np.reshape(x.take(3, 1), [batch_size, 1]),
                user_job: np.reshape(x.take(4, 1), [batch_size, 1]),
                movie_id: np.reshape(x.take(1, 1), [batch_size, 1]),
                movie_categories: categories,  # x.take(6,1)
                movie_titles: titles,  # x.take(5,1)
                title_size: title_sizes,
                targets: np.reshape(y, [batch_size, 1]),
                dropout_keep_prob: dropout_keep,  # dropout_keep
                lr: learning_rate
            }

            # step, train_loss, summaries, _ = sess.run([global_step, loss, train_summary_op, train_op], feed)
            step, train_loss, _ = sess.run([global_step, loss, train_op], feed)

            losses['train'].append(train_loss)
            # train_summary_writer.add_summary(summaries, step)

            # Show every <show_every_n_batches> batches
            if (epoch_i * (len(train_X) // batch_size) + batch_i) % show_every_n_batches == 0:
                time_str = datetime.datetime.now().isoformat()
                print('{}: Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    time_str,
                    epoch_i,
                    batch_i,
                    (len(train_X) // batch_size),
                    train_loss))

    # Save Model
    saver.save(sess, save_dir)
    print('Model Trained and Saved')



