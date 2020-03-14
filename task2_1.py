from pyspark import SparkContext
from math import sqrt
import time

sc = SparkContext()
start_time = time.time()

def split_and_int(s):
    t = s.split(",")
    t[2] = float(t[2])
    return tuple(t)


def find_similarity(active_item_dict, item_dict):
    """
    :param active_item_dict: { user_id: rating ... }
    :param item_dict: {user_id: rating ... }
    :return: pearson similarity of these two items
    """

    '''for key, val in active_item_dict.items():
        print("active_item_dict-",key,",",val)

    for key, val in item_dict.items():
        print("item_dict-",key,",",val)'''

    a = [active_item_dict[key] for key in active_item_dict if key in item_dict]
    b = [item_dict[key] for key in item_dict if key in active_item_dict]

    if not a or not b:
        return 0

    active_item_rating_avg = sum(a) / len(a)
    # print("activeitem_avg = ",active_item_rating_avg)
    item_rating_avg = sum(b) / len(b)
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
        return numerator/denominator
    else:
        return numer_sum/len(similarity_list)

# ----------------------------------------------------------------------------------------------------------------------


train_rdd = sc.textFile("yelp_train.csv").filter(lambda s: s.startswith('user_id') is False) \
    .map(split_and_int) # user_id, item_id, rating

train_rdd_gbuser = train_rdd.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().persist() #user,[(item, rating)...]
train_rdd_gbuser_dict = dict(train_rdd_gbuser.collect())
train_rdd_gbitem = train_rdd.map(lambda x: (x[1], (x[0], x[2]))).groupByKey().persist()
train_rdd_gbitem_dict = dict(train_rdd_gbitem.collect())

test_rdd = sc.textFile("yelp_val.csv").filter(lambda s: s.startswith('user_id') is False).map(lambda s: split_and_int(s))

active_users_items = test_rdd.map(lambda t: (t[0], t[1]))

def find_predictions(actives, train_rdd_gbitem_dict, train_rdd_gbuser_dict):
    """
    I/P = (active_user, active_item)
    :return: (active_user, active_item, pred
    """
    active_user = actives[0]
    active_item = actives[1]

    # all user, ratings that have rated active_item
    if active_item in train_rdd_gbitem_dict:
        active_item_dict = dict(list(train_rdd_gbitem_dict[active_item])) # {user: rating, user: rating, ...}
    else:
        # item not found in training set
        # new item problem.
        average_of_user_list = list(train_rdd_gbuser_dict[active_user])
        average_of_user = sum([x[1] for x in average_of_user_list]) / len(average_of_user_list)
        return active_user, active_item, average_of_user


    # user rated items - all (item, ratings) that the user has rated
    if active_user in train_rdd_gbuser_dict:
        active_user_rated_items = list(train_rdd_gbuser_dict[active_user]) # [(item, rating), (item, rating), ...]
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
    similarity_list.sort(key=lambda x: x[3],reverse=True)
    similarity_list = similarity_list[:len(similarity_list) // 3]
    pred_rating = find_weighted_average(similarity_list)

    # for i in similarity_list:
        # print(i)
    # print("Pred-rating: ", pred_rating)

    return active_user, active_item, pred_rating


result_rdd = active_users_items.map(lambda x: find_predictions(x, train_rdd_gbitem_dict, train_rdd_gbuser_dict))

print("Test rdd count = ", test_rdd.count())
print("result rdd count = ", result_rdd.count())

# not accounted for new user, new item cold start problem.
# find out mean error. :/ (hopeful to be not bad)

print("Time taken: ", time.time() - start_time,"s")