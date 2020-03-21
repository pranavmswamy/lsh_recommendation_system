from pyspark import SparkContext
from math import sqrt
from sys import argv
from time import time

sc = SparkContext()
sc.setLogLevel("ERROR")


# -----------------------------------------------------------------------------------------------------------------------


def split_and_int(s):
    t = s.split(",")
    t[2] = float(t[2])
    return tuple(t)


def split_and_int_tup(s):
    t = s.split(",")
    return t[0], t[1]


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

    user_avg = numer_sum / len_sim_list
    denominator += num_empty_cells
    numerator = numerator + num_empty_cells * user_avg
    return numerator / denominator


def find_predictions(actives, train_rdd_gbitem_dict, train_rdd_gbuser_dict, num_items):
    active_user = actives[0]
    active_item = actives[1]

    if active_user not in train_rdd_gbuser_dict and active_item not in train_rdd_gbitem_dict:
        # new user, new item problem
        return (active_user, active_item), 2.5

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
        # user not found in training set; new user problem.
        return (active_user, active_item), train_rdd_gbitem_dict[active_item][1]

    similarity_list = []
    for item, rating in active_user_rated_items:
        item_dict = dict(train_rdd_gbitem_dict[item][0])
        item_avg = train_rdd_gbitem_dict[item][1]
        similarity = find_similarity(dict(active_item_dict), active_item_avg, dict(item_dict), item_avg)
        similarity_list.append((rating, similarity))

    pred_rating = find_weighted_average(similarity_list, num_items)
    return (active_user, active_item), pred_rating


def get_prediction_using_cfrs():
    start_time = time()
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
    tcount = test_rdd.count()
    pred_rdd = test_rdd.map(
        lambda x: find_predictions(x, train_rdd_gbitem_dict, train_rdd_gbuser_dict, num_items))
    print("Time taken: ", time() - start_time, "s")
    return pred_rdd.collect()


def write_to_file(pred):
    with open(str(argv[3]), "w") as file:
        file.write("user_id, business_id, prediction")
        for row in pred:
            file.write(str("\n" + row[0][0] + "," + row[0][1] + "," + str(row[1])))
        file.close()
    print("Finished writing to file")


# ----------------------------------------------------------------------------------------------------------------------

prediction = get_prediction_using_cfrs()
write_to_file(prediction)