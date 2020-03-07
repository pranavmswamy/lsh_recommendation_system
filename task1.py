from pyspark import SparkContext

sc = SparkContext()

# -----------------------------
def hash_to_32_bit_nums(businesses_itr):
    businesses = list(businesses_itr)
    return [hash(business_id) % (2**32)-1 for business_id in businesses]


# -----------------------------

rdd = sc.textFile("yelp_train.csv").filter(lambda s: s.startswith('user_id') is False)\
    .map(lambda s: tuple(s.split(",")[:2]))

group_by_user_rdd = rdd.groupByKey().mapValues(hash_to_32_bit_nums)

for i in group_by_user_rdd.take(5):
    print(i)

user_id_rdd = rdd.map(lambda r: r[0]).distinct()
business_id_rdd = rdd.map(lambda r: r[1]).distinct()


# print(f'User-RDD-Count={user_id_rdd.count()}')
# print(f'Biz-RDD-Count={business_id_rdd.count()}')

# for i in rdd.take(5):
   # print(i)