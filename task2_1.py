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


def find_similarity(active_item_dict, item_dict):
    """
    :param active_item_dict: { user_id: rating ... }
    :param item_dict: {user_id: rating ... }
    :return: pearson similarity of these two items
    """
    # a = [active_item_dict[key] for key in active_item_dict if key in item_dict]
    # b = [item_dict[key] for key in item_dict if key in active_item_dict]

    a = sum(active_item_dict.values())
    b = sum(item_dict.values())

    if a == 0 or b == 0:
        return 0

    active_item_rating_avg = a / len(active_item_dict)
    # print("activeitem_avg = ",active_item_rating_avg)
    item_rating_avg = b / len(item_dict)
    # print("item_avg = ",item_rating_avg)

    for user_id in active_item_dict:
        active_item_dict[user_id] -= active_item_rating_avg
    for user_id in item_dict:
        item_dict[user_id] -= item_rating_avg

    pearson_numerator = 0

    for user_id in active_item_dict:
        if user_id in item_dict:
            pearson_numerator += (active_item_dict[user_id] * item_dict[user_id])

    # print("Pearson_num = ", pearson_numerator)

    for user_id in active_item_dict:
        active_item_dict[user_id] *= active_item_dict[user_id]
    for user_id in item_dict:
        item_dict[user_id] *= item_dict[user_id]

    pearson_denom_left = sum([active_item_dict[key] for key in active_item_dict if key in item_dict])
    # print("Denomleft = ", pearson_denom_left)
    pearson_denom_right = sum([item_dict[key] for key in item_dict if key in active_item_dict])
    # print("Denomright = ", pearson_denom_right)
    pearson_denom = sqrt(pearson_denom_left * pearson_denom_right)

    if pearson_numerator == 0 or pearson_denom == 0:
        return 0
    sim = pearson_numerator / pearson_denom
    # print("Sim=",sim )
    return sim


def find_weighted_average(similarity_list):
    numerator = 0
    denominator = 0
    numer_sum = 0
    for row in similarity_list:
        numerator += (row[2] * row[3])
        denominator += abs(row[3])
        numer_sum += row[2]

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

    # all user, ratings that have rated active_item
    if active_item in train_rdd_gbitem_dict:
        active_item_dict = dict(list(train_rdd_gbitem_dict[active_item]))  # {user: rating, user: rating, ...}
    else:
        # item not found in training set
        # new item problem.
        average_of_user_list = list(train_rdd_gbuser_dict[active_user])
        average_of_user = sum([x[1] for x in average_of_user_list]) / len(average_of_user_list)
        return active_user, active_item, average_of_user

    # user rated items - all (item, ratings) that the user has rated
    if active_user in train_rdd_gbuser_dict:
        active_user_rated_items = list(train_rdd_gbuser_dict[active_user])  # [(item, rating), (item, rating), ...]
    else:
        # user not found in training set
        # new user problem.
        average_of_item_list = list(train_rdd_gbitem_dict[active_item])
        average_of_item = sum(x[1] for x in average_of_item_list) / len(average_of_item_list)
        return active_user, active_item, average_of_item

    similarity_list = []

    for item, rating in active_user_rated_items:
        item_dict = dict(list(train_rdd_gbitem_dict[item]))
        similarity = find_similarity(dict(active_item_dict), dict(item_dict))
        similarity_list.append((active_item, item, rating, similarity))

    # Have obtained similarity list for active item and item from the above code.
    # Filter according to a top 'N' items and then take avg rating.
    similarity_list.sort(key=lambda x: x[3], reverse=True)
    similarity_list = similarity_list[:len(similarity_list) // 2]
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

train_rdd_gbuser = train_rdd.map(lambda x: (x[0], (x[1], x[2]))).groupByKey()  # user,[(item, rating)...]
train_rdd_gbuser_dict = dict(train_rdd_gbuser.collect())

t2 = time.time()
print("Time taken to map and group train_rdduser = ", t2 - t1, "s")

train_rdd_gbitem = train_rdd.map(lambda x: (x[1], (x[0], x[2]))).groupByKey()
train_rdd_gbitem_dict = dict(train_rdd_gbitem.collect())


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

result_rdd = test_rdd.join(pred_rdd).map(lambda x: (x[1][0] - x[1][1])**2)
result = result_rdd.reduce(lambda x,y: x + y)
print("Result(Numerator) = ", result)
x = result / pred_count

t7 = time.time()
print("Time taken to calc rmse = ", t7 - t6, "s")

rmse = sqrt(x)

print("RMSE = ", rmse)

#  print(result_rdd.take(5))

# test_rdd_dict = dict(test_rdd.collect())
# pred_rdd_dict = dict(pred_rdd.collect())

# testrdd - (active_user, active_item, rating)
# resultrdd - (Active_user, active_item, pred_rating)



'''summation = 0
for i in test_rdd_dict:
    if i in pred_rdd_dict:
        summation += (test_rdd_dict[i] - pred_rdd_dict[i])**2

    result = summation / len(test_rdd_dict)

    rmse = sqrt(result)

print("RMSE = ", rmse)'''

# not accounted for new user & new item cold start problem.
# find out mean error. :/ (hopeful to be not bad)

print("Time taken: ", time.time() - start_time, "s")
