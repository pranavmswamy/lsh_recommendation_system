from pyspark import SparkContext
from math import sqrt
import time

sc = SparkContext()
start_time = time.time()


def split_and_int(s):
    t = s.split(",")
    t[2] = float(t[2])
    return tuple(t)


def split_and_int_tup(s):
    t = s.split(",")
    t[2] = float(t[2])
    return (t[0], t[1]), t[2]


def find_similarity(active_item_dict, active_item_avg, item_dict, item_avg):
    """
    :param item_avg:
    :param active_item_avg:
    :param active_item_dict: { user_id: rating ... }
    :param item_dict: {user_id: rating ... }
    :return: pearson similarity of these two items
    """
    pearson_numerator = 0
    pearson_denom_left = 0
    pearson_denom_right = 0

    for user_id in active_item_dict:
        if user_id in item_dict:
            pearson_numerator += ((active_item_dict[user_id] - active_item_avg) * (item_dict[user_id] - item_avg))
            pearson_denom_left += (active_item_dict[user_id] - active_item_avg) ** 2
            pearson_denom_right += (item_dict[user_id] - item_avg) ** 2

    pearson_denom = sqrt(pearson_denom_left * pearson_denom_right)

    if pearson_denom == 0:
        return 0
    else:
        return pearson_numerator / pearson_denom


def find_weighted_average(similarity_list):
    numerator = 0
    denominator = 0
    numer_sum = 0
    for row in similarity_list:
        numerator += (row[0] * row[1])
        denominator += abs(row[1])
        numer_sum += row[0]

    if denominator != 0:
        return numerator / denominator
    else:
        return numer_sum / len(similarity_list)


def find_predictions(actives, train_rdd_gbitem_dict, train_rdd_gbuser_dict):
    """
    I/P = (active_user, active_item)
    :return: (active_user, active_item, pred
    """
    active_user = actives[0][0]
    active_item = actives[0][1]

    # -----------------------------------
    # train_rdd_gbitem_dict = (item, ([(user,r),(user,r)...],avg_of_item))
    # train_rdd_gbuser_dict = (user, [(item,r),(item,r)...]

    # all user, ratings that have rated active_item
    if active_item in train_rdd_gbitem_dict:
        active_item_avg = train_rdd_gbitem_dict[active_item][1]
        active_item_dict = dict(train_rdd_gbitem_dict[active_item][0])  # {user: rating, user: rating, ...}
    else:
        # item not found in training set
        # new item problem.
        average_of_user_list = train_rdd_gbuser_dict[active_user]
        average_of_user = sum([x[1] for x in average_of_user_list]) / len(average_of_user_list)
        return (active_user, active_item), average_of_user

    # user rated items - all (item, ratings) that the user has rated
    if active_user in train_rdd_gbuser_dict:
        active_user_rated_items = train_rdd_gbuser_dict[active_user]  # [(item, rating), (item, rating), ...]
    else:
        # user not found in training set
        # new user problem.
        return (active_user, active_item), train_rdd_gbitem_dict[active_item][1]

    similarity_list = []
    for item, rating in active_user_rated_items:
        item_dict = dict(train_rdd_gbitem_dict[item][0])
        item_avg = train_rdd_gbitem_dict[item][1]
        similarity = find_similarity(dict(active_item_dict), active_item_avg, dict(item_dict), item_avg)
        similarity_list.append((rating, similarity))

    # Have obtained similarity list for active item and item from the above code.
    # Filter according to a top 'N' items and then take avg rating.
    similarity_list.sort(key=lambda x: x[1], reverse=True)
    similarity_list = similarity_list[:len(similarity_list) // 4]
    pred_rating = find_weighted_average(similarity_list)

    # for i in similarity_list:
    # print(i)
    # print("Pred-rating: ", pred_rating)

    return (active_user, active_item), pred_rating

# ----------------------------------------------------------------------------------------------------------------------

train_rdd = sc.textFile("yelp_train.csv").filter(lambda s: s.startswith('user_id') is False) \
    .map(split_and_int)  # user_id, item_id, rating

t1 = time.time()
print("Time taken to load train_rdd = ", t1 - start_time, "s")

train_rdd_gbuser = train_rdd.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().mapValues(
    lambda x: list(x))  # user,[(item, rating)...]
train_rdd_gbuser_dict = dict(train_rdd_gbuser.collect())

t2 = time.time()
print("Time taken to map and group train_rdduser = ", t2 - t1, "s")

train_rdd_gbitem = train_rdd.map(lambda x: (x[1], (x[0], x[2]))).groupByKey().mapValues(lambda x: list(x)).mapValues(
    lambda l: (l, sum([i[1] for i in l]) / len(l)))
train_rdd_gbitem_dict = dict(train_rdd_gbitem.collect())

# print("\n",train_rdd_gbitem_dict.items())

t3 = time.time()
print("Time taken to map and group train_rdditem = ", t3 - t2, "s")

test_rdd = sc.textFile("yelp_val.csv").filter(lambda s: s.startswith('user_id') is False).map(
    lambda s: split_and_int_tup(s)).persist()
tcount = test_rdd.count()

t4 = time.time()
print("Time taken to load test_rdd = ", t4 - t3, "s")
# active_users_items = test_rdd.map(lambda t: (t[0], t[1]))
pred_rdd = test_rdd.map(lambda x: find_predictions(x, train_rdd_gbitem_dict, train_rdd_gbuser_dict)).persist()

t5 = time.time()
print("Time taken to generate preds-main step = ", t5 - t4, "s")

pred_count = pred_rdd.count()
print("Test rdd count = ", test_rdd.count())
print("result rdd count = ", pred_count)

t6 = time.time()
print("Time taken to generate rdd counts = ", t6 - t5, "s")

result_rdd = test_rdd.join(pred_rdd).map(lambda x: (x[1][0] - x[1][1]) ** 2)
result = result_rdd.reduce(lambda x, y: x + y)
print("Result(Numerator) = ", result)
x = result / pred_count

t7 = time.time()
print("Time taken to calc rmse = ", t7 - t6, "s")

rmse = sqrt(x)
print("RMSE = ", rmse)

# not accounted for new user & new item cold start problem.
# find out mean error. :/ (hopeful to be not bad)

print("Time taken: ", time.time() - start_time, "s")