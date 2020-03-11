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


def lsh(sig_key_column):
    '''
    generate b hash functions
    run them through each bth part of the column
    generate (b_num, hash_val), b_id) for each set of rows in a band.
    group by key
    return ((band_num, hashvalue), (b_id))
    :param sig_column:
    :return:
    '''
    c = 4294967539

    r = 5
    band_hash_list = []
    sig_column = sig_key_column[1]

    counter = 0
    b = 0
    for counter in range(0,len(sig_column),r):
        left_idx = counter
        right_idx = counter + r
        b += 1

        combined_str = ""
        for num in range(left_idx, right_idx):
            combined_str = combined_str + str(sig_column[num])
        band_row_hash = (h1(int(combined_str)) + b * h2(int(combined_str))) % c
        band_hash_list.append((str(b)+ str(band_row_hash), sig_key_column[0]))
        # returns list of tuples

    return band_hash_list

# -----------------------------

rdd = sc.textFile("yelp_train.csv").filter(lambda s: s.startswith('user_id') is False) \
    .map(lambda s: tuple(s.split(",")[-2::-1])) # business_id, user_id

rdd_by_business = rdd.groupByKey().mapValues(hash_to_32_bit_nums)

sig_matrix_rdd = rdd_by_business.mapValues(convert_to_sig_matrix)

# checked upto here, working properly - sig matrix is correct. Tested with two businesses on Piazza, got a val of 0.65 (Piazza val - 0.66).

lsh_rdd = sig_matrix_rdd.map(lsh)
lsh_rdd = lsh_rdd.flatMap(lambda list: list).groupByKey().mapValues(lambda l: list(l))
lsh_rdd = lsh_rdd.filter(lambda x: len(x[1]) > 1)

sim_items_rdd = lsh_rdd.map(lambda x: tuple(sorted(x[1]))).distinct()

a = sim_items_rdd.collect()

for i in a:
    print("\n",i)