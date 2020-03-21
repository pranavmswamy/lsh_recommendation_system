from pyspark import SparkContext
from math import sqrt
from sys import argv
from sklearn.metrics import mean_squared_error
from time import time
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from json import loads

sc = SparkContext()
sc.setLogLevel("ERROR")

# CF BASED RF HELPER METHODS

def split_and_int(s):
    t = s.split(",")
    t[2] = float(t[2])
    return tuple(t)


def split_and_int_tup(s):
    t = s.split(",")
    t[2] = float(t[2])
    return (t[0], t[1]), t[2]


def find_similarity(active_item_dict, active_item_avg, item_dict, item_avg):
    pearson_numerator = 0
    pearson_denom_left = 0
    pearson_denom_right = 0

    for user_id in active_item_dict:
        if user_id in item_dict:
            activeMinusavg = active_item_dict[user_id] - active_item_avg
            itemMinusavg = item_dict[user_id] - item_avg
            pearson_numerator += (activeMinusavg * itemMinusavg)
            pearson_denom_left += activeMinusavg ** 2
            pearson_denom_right += itemMinusavg ** 2

    pearson_denom = sqrt(pearson_denom_left * pearson_denom_right)

    if pearson_denom == 0:
        return 0
    else:
        return pearson_numerator / pearson_denom


def find_weighted_average(similarity_list, num_items):
    numerator = 0
    denominator = 0
    numer_sum = 0

    if not similarity_list:
        return 3.5

    len_sim_list = len(similarity_list)
    num_empty_cells = num_items - len_sim_list

    for row in similarity_list:
        numerator += (row[0] * row[1])
        denominator += abs(row[1])
        numer_sum += row[0]

    user_avg = numer_sum/len_sim_list

    denominator += num_empty_cells
    numerator = numerator + num_empty_cells * user_avg

    return numerator / denominator


def find_predictions(actives, train_rdd_gbitem_dict, train_rdd_gbuser_dict, num_items):
    active_user = actives[0][0]
    active_item = actives[0][1]

    if active_user not in train_rdd_gbuser_dict and active_item not in train_rdd_gbitem_dict:
        return (active_user, active_item), 2.5

    # all user, ratings that have rated active_item
    if active_item in train_rdd_gbitem_dict:
        active_item_avg = train_rdd_gbitem_dict[active_item][1]
        active_item_dict = dict(train_rdd_gbitem_dict[active_item][0])  # {user: rating, user: rating, ...}
    else:
        # item not found in training set; new item problem.
        average_of_user_list = train_rdd_gbuser_dict[active_user]
        average_of_user = sum([x[1] for x in average_of_user_list]) / len(average_of_user_list)
        return (active_user, active_item), average_of_user

    # user rated items - all (item, ratings) that the user has rated
    if active_user in train_rdd_gbuser_dict:
        active_user_rated_items = train_rdd_gbuser_dict[active_user]  # [(item, rating), (item, rating), ...]
    else:
        # user not found in training set, new user problem.
        return (active_user, active_item), train_rdd_gbitem_dict[active_item][1]

    similarity_list = []
    for item, rating in active_user_rated_items:
        item_dict = dict(train_rdd_gbitem_dict[item][0])
        item_avg = train_rdd_gbitem_dict[item][1]
        similarity = find_similarity(dict(active_item_dict), active_item_avg, dict(item_dict), item_avg)
        similarity_list.append((rating, similarity))

    pred_rating = find_weighted_average(similarity_list, num_items)
    return (active_user, active_item), pred_rating

# MODEL BASED RC HELPER METHODS

def filter_user_data(user_data):
    return (user_data["user_id"],
            (user_data["review_count"], len(user_data["friends"]), user_data["average_stars"]))


def filter_business_data(business_data):
    return (business_data["business_id"],
            (business_data["city"], business_data["stars"], business_data["review_count"],
             business_data["categories"]))


def unpack_user_joined_tuples(row):
    business_id = row[1][0][0]
    user_id = row[0]
    rating = row[1][0][1]
    review_count = row[1][1][0]
    num_friends = row[1][1][1]
    average_stars_of_user = row[1][1][2]
    return (business_id, (user_id, rating, review_count, num_friends, average_stars_of_user))
    # ('Ytjon2aVUm09CMdfGJxZYg', (('6qL7HYCVN1E0vLfa-S_VlQ', 3.0), (636, 6622, 3.97)))


def unpack_business_joined_tuples(row):
    # ('1sKRc9vFaZ_dLQGPE_P7Dg', (('Xxvz5g67eaCr3emnkY5M6w', 2.0, 2075, 15334, 3.73),
    # ('Westlake', 3.0, 69, 1, 'Nail Salons, Beauty & Spas, Day Spas')))
    business_id = row[0]
    user_id = row[1][0][0]
    rating_y = row[1][0][1]
    user_review_count = row[1][0][2]
    num_friends = row[1][0][3]
    avg_stars_of_user = row[1][0][4]

    city = row[1][1][0]
    avg_stars_business = row[1][1][1]
    business_review_count = row[1][1][2]
    categories = row[1][1][3]

    # exculding categories for now. Could be important
    return user_id, business_id, user_review_count, num_friends, avg_stars_of_user, city, avg_stars_business, \
           business_review_count, rating_y

def get_prediction_using_cfrs():
    cf_start_time = time()
    train_rdd = sc.textFile(str(argv[1])).filter(lambda s: s.startswith('user_id') is False) \
        .map(split_and_int)  # user_id, item_id, rating
    num_items = train_rdd.map(lambda x: x[1]).distinct().count()
    train_rdd_gbuser = train_rdd.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().mapValues(
        lambda x: list(x))  # user,[(item, rating)...]
    train_rdd_gbuser_dict = dict(train_rdd_gbuser.collect())
    train_rdd_gbitem = train_rdd.map(lambda x: (x[1], (x[0], x[2]))).groupByKey().mapValues(
        lambda x: list(x)).mapValues(
        lambda l: (l, sum([i[1] for i in l]) / len(l)))
    train_rdd_gbitem_dict = dict(train_rdd_gbitem.collect())
    test_rdd = sc.textFile(str(argv[2])).filter(lambda s: s.startswith('user_id') is False).map(
        lambda s: split_and_int_tup(s)).persist()
    t_count = test_rdd.count()
    pred_rdd = test_rdd.map(
        lambda x: find_predictions(x, train_rdd_gbitem_dict, train_rdd_gbuser_dict, num_items)).persist()
    pred_count = pred_rdd.count()
    result_rdd = test_rdd.join(pred_rdd).map(lambda x: (x[1][0] - x[1][1]) ** 2)
    result = result_rdd.reduce(lambda x, y: x + y)
    x = result / pred_count
    rmse = sqrt(x)
    print("CF RMSE =", rmse, " time taken = ", time() - cf_start_time, "s")
    return pred_rdd.collectAsMap()

# ----------------------------------------------------------------------------------------------------------------------

def get_prediction_using_mbrc():
    start_time = time()
    # Loading yelp_train, removing header and mapping it into tuples. (user_id, biz_id, rating)
    yelp_train_rdd = sc.textFile(str(argv[1] + "/yelp_train.csv")).filter(
        lambda s: s.startswith('user_id') is False).map(split_and_int)
    # Loading yelp_val, removing header and mapping it into tuples. (user_id, biz_id)
    yelp_val_rdd = sc.textFile(str(argv[2])).filter(lambda s: s.startswith('user_id') is False).map(
        split_and_int)
    yelp_val_inorder = yelp_val_rdd.collect()
    # Loading user.json and filtering out extraneous data
    users_rdd = sc.textFile(str(argv[1] + "user-002.json")).map(loads).map(filter_user_data).persist()
    # Loading business.json and filtering out extraneous data
    businesses_rdd = sc.textFile(str(argv[1] + "business.json")).map(loads).map(filter_business_data).persist()
    # Joining yelp_train with user.json
    yelp_train_rdd = yelp_train_rdd.map(lambda x: (x[0], (x[1], x[2]))).join(users_rdd).map(unpack_user_joined_tuples)
    # Joining yelp_val with user.json
    yelp_val_rdd = yelp_val_rdd.map(lambda x: (x[0], (x[1], x[2]))).join(users_rdd).map(unpack_user_joined_tuples)
    # Joining yelp_train with business.json
    yelp_train_rdd = yelp_train_rdd.join(businesses_rdd).map(unpack_business_joined_tuples)
    # Joining yelp_val with business.json
    yelp_val_rdd = yelp_val_rdd.join(businesses_rdd).map(unpack_business_joined_tuples)
    # collecting list of tuples for yelp_train
    yelp_train_list = yelp_train_rdd.collect()
    # collecting list of tuples for yelp_val
    yelp_val_list = yelp_val_rdd.collect()
    # unpersisting the rdds
    users_rdd.unpersist()
    businesses_rdd.unpersist()

    # converting yelp_train list to df
    yelp_train = pd.DataFrame(data=yelp_train_list,
                              columns=['user_id', 'business_id', 'user_review_count', 'num_friends',
                                       'avg_stars_of_user', 'city', 'avg_stars_business',
                                       'business_review_count', 'rating_y'])
    # converting yelp_val list to df
    yelp_val = pd.DataFrame(data=yelp_val_list, columns=['user_id', 'business_id', 'user_review_count', 'num_friends',
                                                         'avg_stars_of_user', 'city', 'avg_stars_business',
                                                         'business_review_count', 'rating_y'])

    yelp_val_copy = yelp_val.copy()
    yelp_val_copy.drop(columns=['user_review_count', 'num_friends',
                                'avg_stars_of_user', 'city', 'avg_stars_business',
                                'business_review_count'], inplace=True)

    numerical_vars = ['user_review_count', 'num_friends', 'avg_stars_of_user', 'avg_stars_business',
                      'business_review_count', 'rating_y']

    categorical_vars = ['user_id', 'business_id', 'city']

    # transforming categorical vars of yelp_train
    for col in yelp_train.columns:
        if yelp_train[col].dtype == 'object':
            label = LabelEncoder()
            label.fit(list(yelp_train[col].values))
            yelp_train[col] = label.transform(list(yelp_train[col].values))

    # transforming categorical vars of yelp_val
    for col in yelp_val.columns:
        if yelp_val[col].dtype == 'object':
            label = LabelEncoder()
            label.fit(list(yelp_val[col].values))
            yelp_val[col] = label.transform(list(yelp_val[col].values))

    # train test split.
    X_train, y_train = yelp_train.iloc[:, :-1], yelp_train.iloc[:, -1]
    X_test, y_test = yelp_val.iloc[:, :-1], yelp_val.iloc[:, -1]

    # building model and printing rmse
    yelp_train_dmatrix = xgb.DMatrix(data=X_train, label=y_train)
    xgb_model = xgb.XGBRegressor(learning_rate=0.1, n_estimators=250)
    xgb_model.fit(X_train, y_train)
    predictions = xgb_model.predict(X_test)

    rmse = sqrt(mean_squared_error(y_test, predictions))
    print("RMSE = ", rmse)
    print("Time taken: ", time() - start_time, "s")

    yelp_val_copy["prediction"] = predictions
    yelp_val_copy.set_index(['user_id', 'business_id'], inplace=True)


def write_to_file(yelp_val_copy, yelp_val_inorder):
    # Writing to file
    with open(str(argv[3]), "w") as file:
        file.write("user_id, business_id, prediction")
        for row in yelp_val_inorder:
            file.write(str("\n" + row[0] + "," + row[1] + "," + str(yelp_val_copy.loc[row[0], row[1]]['prediction'])))
        file.close()

# ----------------------------------------------------------------------------------------------------------------------

