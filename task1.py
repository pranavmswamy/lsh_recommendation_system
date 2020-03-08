from pyspark import SparkContext

sc = SparkContext()


# -----------------------------
def hash_to_32_bit_nums(businesses_itr):
    businesses = list(businesses_itr)
    return [hash(business_id) % 4294967295 for business_id in businesses]


def h1(x):
    a = 2976840907
    b = 3727260824
    c = 4294967539
    return (a * x + b) % c


def h2(x):
    a = 2100863184
    b = 634485346
    c = 4294967539
    return (a * x + b) % c


def convert_to_sig_matrix(businesses):
    c = 4294967539
    sig_column = []
    num_hash_fns = 100

    for i in range(num_hash_fns):
        min_hash = c + 29

        for id in businesses:
            hash_code = (h1(id) + i * h2(id)) % c
            if hash_code < min_hash:
                min_hash = hash_code

        sig_column.append(min_hash)

    return sig_column


# -----------------------------

rdd = sc.textFile("yelp_train.csv").filter(lambda s: s.startswith('user_id') is False) \
    .map(lambda s: tuple(s.split(",")[-2::-1])) # business_id, user_id

rdd_by_business = rdd.groupByKey().mapValues(hash_to_32_bit_nums)

sig_matrix_rdd = rdd_by_business.mapValues(convert_to_sig_matrix)

# checked upto here, working properly - sig matrix is correct. Tested with two businesses on Piazza, got a val of 0.65 (Piazza val - 0.66).

