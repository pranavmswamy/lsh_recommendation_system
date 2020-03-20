import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from math import sqrt
from pyspark import SparkContext
from json import loads
from time import time
import matplotlib.pyplot as plt
from sys import argv

# -----------------------------------
def split_and_int(s):
    t = s.split(",")
    t[2] = float(t[2])
    return tuple(t)


def filter_user_data(user_data):
    # returning len(friends) for now. If RMSE is low, see what can be done by returning entire list.
    return (user_data["user_id"],
            (user_data["review_count"], len(user_data["friends"]), user_data["average_stars"]))


def filter_business_data(business_data):
    # removing attributes for now.
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


# ------------------------------------
# initialising SparkContext
sc = SparkContext()
sc.setLogLevel("ERROR")

# recording start time
start_time = time()

argv = ["", "", "", "task2_2pred.csv", ""]

# Loading yelp_train, removing header and mapping it into tuples. (user_id, biz_id, rating)
yelp_train_rdd = sc.textFile(str(argv[1]+"yelp_train.csv")).filter(lambda s: s.startswith('user_id') is False).map(split_and_int)

# Loading yelp_val, removing header and mapping it into tuples. (user_id, biz_id)
yelp_val_rdd = sc.textFile(str("yelp_val.csv")).filter(lambda s: s.startswith('user_id') is False).map(split_and_int)
yelp_val_inorder = yelp_val_rdd.collect()

# Loading user.json and filtering out extraneous data
users_rdd = sc.textFile(str(argv[1]+"user-002.json")).map(loads).map(filter_user_data).persist()

# Loading business.json and filtering out extraneous data
businesses_rdd = sc.textFile(str(argv[1]+"business.json")).map(loads).map(filter_business_data).persist()

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

'''for i in range(5):
    print(yelp_train_list[i])

for i in range(5):
    print(yelp_val_list[i])'''

# converting yelp_train list to df
yelp_train = pd.DataFrame(data=yelp_train_list, columns=['user_id', 'business_id', 'user_review_count', 'num_friends',
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
yelp_val_copy.to_csv("./yelp_val_copy.csv")

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
xgb_model = xgb.XGBRegressor(learning_rate=0.1,n_estimators=250)
xgb_model.fit(X_train, y_train)
predictions = xgb_model.predict(X_test)

rmse = sqrt(mean_squared_error(y_test, predictions))
print("RMSE = ", rmse)
print("Time taken: ", time() - start_time, "s")

# Writing to file
yelp_val_copy["prediction"] = predictions
yelp_val_copy.set_index(['user_id', 'business_id'], inplace=True)

with open(str(argv[3]), "w") as file:
    file.write("user_id, business_id, prediction")
    for row in yelp_val_inorder:
        file.write(str("\n" + row[0] + "," + row[1] + "," + str(yelp_val_copy.loc[row[0], row[1]]['prediction'])))

    # for index, row in yelp_val_copy.iterrows():
        # file.write(str("\n" + row['user_id']+ "," + row['business_id'] + "," + str(row['prediction'])))
    file.close()