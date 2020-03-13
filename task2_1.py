from pyspark import SparkContext
from math import sqrt

sc = SparkContext()

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
    print("Sim=",sim )
    return sim


train_rdd = sc.textFile("yelp_train.csv").filter(lambda s: s.startswith('user_id') is False) \
    .map(split_and_int) # user_id, item_id, rating

train_rdd_gbyuser = train_rdd.map(lambda x: (x[0], (x[1], x[2]))).groupByKey().persist() #user,[item,...]
train_rdd_gbitem = train_rdd.map(lambda x: (x[1], (x[0], x[2]))).groupByKey().persist()

test_rdd = sc.textFile("yelp_val.csv").filter(lambda s: s.startswith('user_id') is False).map(lambda s: tuple(s.split(",")))

active_users_items = test_rdd.map(lambda t: (t[0], t[1])).collect()

similarity_list = []
for (active_user, active_item) in active_users_items:
    # all user, ratings that have rated active_item
    active_item_dict = dict(list(train_rdd_gbitem.filter(lambda k: k[0] == active_item).collect()[0][1]))


    # user rated items - all item, ratings that the user has rated
    active_user_rated_items = list(train_rdd_gbyuser.filter(lambda k: k[0] == active_user).collect()[0][1])
    print(active_item_dict)

    for (item, rating) in active_user_rated_items:
        item_dict = dict(list(train_rdd_gbitem.filter(lambda k: k[0] == item).collect()[0][1]))
        similarity = find_similarity(dict(active_item_dict), item_dict)
        similarity_list.append((active_item, item, similarity))

for i in similarity_list:
    print(i)