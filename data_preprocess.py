import pandas as pd
import re
import pickle
import numpy as np


def load_data():
    # 读取user数据
    user_title = ['UserID', 'Gender', 'Age', 'JobID', 'Zip-code']
    users = pd.read_table('./ml-1m/users.dat', sep='::', header=None, names=user_title, engine='python')
    users = users.filter(regex='UserID|Gender|Age|JobID')
    users_orig = users.values
    # 性别和年龄变成数字
    gender_map = {'F': 0, 'M': 1}
    users['Gender'] = users['Gender'].map(gender_map)
    age_map = {val: ii for ii, val in enumerate(set(users['Age']))}
    users['Age'] = users['Age'].map(age_map)

    # 读取movie数据集
    movie_title = ['MovieID', 'Title', 'Genres']
    movies = pd.read_table('./ml-1m/movies.dat', sep='::', header=None, names=movie_title, engine='python')
    movies_orig = movies.values
    # 将title中的年份去掉
    pattern = re.compile(r'^(.*)\((\d+)\)$')
    title_map = {val: pattern.match(val).group(1) for ii, val in enumerate(set(movies['Title']))}
    movies['Title'] = movies['Title'].map(title_map)
    # 电影名转数字字典
    title_set = set()
    for val in movies['Title'].str.split():
        title_set.update(val)
    title_set.add('<PAD>')
    title2int = {val: ii for ii, val in enumerate(title_set)}
    # 将电影名转成等长数字列表
    title_count = 15
    title_map = {val: [title2int[row] for row in val.split()] for ii, val in enumerate(set(movies['Title']))}
    for key in title_map:
        for cnt in range(title_count - len(title_map[key])):
            title_map[key].insert(len(title_map[key]) + cnt, title2int['<PAD>'])
    movies['Title'] = movies['Title'].map(title_map)
    # 电影类型转数字字典
    genres_set = set()
    for val in movies['Genres'].str.split('|'):
        genres_set.update(val)
    genres_set.add('<PAD>')
    genres2int = {val: ii for ii, val in enumerate(genres_set)}
    # 电影类型转成等长数字列表
    genres_map = {val: [genres2int[row] for row in val.split('|')] for ii, val in enumerate(set(movies['Genres']))}
    for key in genres_map.keys():
        # 19个类型， 标号0-18， max(*)=18, len(*)=每部电影有几个类型归属， 剩下的补全0
        for cnt in range(max(genres2int.values()) - len(genres_map[key])):
            genres_map[key].insert(len(genres_map[key]) + cnt, genres2int['<PAD>'])
    movies['Genres'] = movies['Genres'].map(genres_map)

    # 专为rnn准备的一列
    title_col = movies['Title']
    len_col = np.array([len(x) - x.count(title2int['<PAD>']) for x in title_col], dtype=np.int32)
    movies['Title_size'] = len_col

    # 读取评分数据
    ratings_title = ['UserID', 'MovieID', 'Ratings', 'Timestamps']
    ratings = pd.read_table('./ml-1m/ratings.dat', sep='::', header=None, names=ratings_title, engine='python')
    ratings = ratings.filter(regex='UserID|MovieID|Ratings')

    # 合并三个表
    data = pd.merge(pd.merge(ratings, users), movies)

    # 将数据分成x和y两张表
    target_fields = ['Ratings']
    features_pd, target_pd = data.drop(target_fields, axis=1), data[target_fields]
    features = features_pd.values
    targets_values = target_pd.values

    return title_count, \
           title_set, \
           genres2int, \
           features, \
           targets_values, \
           ratings, \
           users, \
           movies, \
           data, \
           movies_orig, \
           users_orig


def main(argv=None):
    title_count, title_set, genres2int, features, targets_values, \
        ratings, users, movies, data, movies_orig, users_orig \
        = load_data()

    f = open('preprocess.p', 'wb')
    pickle.dump((
                title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig,
                users_orig), f)
    f.close()

    print(users.head())
    print(movies.head())
    print(ratings.head())


if __name__ == '__main__':
    main()
