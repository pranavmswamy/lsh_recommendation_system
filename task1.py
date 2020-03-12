from pyspark import SparkContext
from itertools import combinations

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
    '''
    c = 4294967539

    r = 5
    band_hash_list = []
    sig_column = sig_key_column[1]

    b = 0
    for counter in range(0, len(sig_column), r):
        left_idx = counter
        right_idx = counter + r
        b += 1

        combined_str = ""
        for num in range(left_idx, right_idx):
            combined_str = combined_str + str(sig_column[num])
        band_row_hash = (h1(int(combined_str)) + b * h2(int(combined_str))) % (100 // r)
        # band_row_hash = int(combined_str) % (100 // r)
        band_hash_list.append((str(b)+ str(band_row_hash), sig_key_column[0]))
        # returns list of tuples

    return band_hash_list


def find_similarity(couple):
    shingles1 = set(users_by_business_id[couple[0]])
    shingles2 = set(users_by_business_id[couple[1]])
    similarity = len(shingles1.intersection(shingles2)) / len(shingles1.union(shingles2))
    # if similarity >= 0.5:
    return (couple[0], couple[1], similarity)
    # else:
        # return ('<0.5')


# -----------------------------

rdd = sc.textFile("yelp_train.csv").filter(lambda s: s.startswith('user_id') is False) \
    .map(lambda s: tuple(s.split(",")[-2::-1])) # business_id, user_id

rdd_by_business_id = rdd.groupByKey().persist()

users_by_business_id = dict(rdd_by_business_id.collect())

rdd_by_business = rdd_by_business_id.mapValues(hash_to_32_bit_nums)

sig_matrix_rdd = rdd_by_business.mapValues(convert_to_sig_matrix)

# checked upto here, working properly - sig matrix is correct. Tested with two businesses on Piazza, got a val of 0.65 (Piazza val - 0.66).

lsh_rdd = sig_matrix_rdd.map(lsh)
lsh_rdd = lsh_rdd.flatMap(lambda list: list).groupByKey().mapValues(lambda l: list(l))
lsh_rdd = lsh_rdd.filter(lambda x: len(x[1]) > 1)

sim_items_candidates_rdd = lsh_rdd.map(lambda x: tuple(sorted(x[1]))).distinct()

sim_items_candidates_rdd = sim_items_candidates_rdd.flatMap(lambda items: combinations(items,2)).map(lambda couple: tuple(sorted(couple))).distinct()

sim_items_candidates_rdd = sim_items_candidates_rdd.map(find_similarity)

similar_candidates = sim_items_candidates_rdd.collect()


for item in similar_candidates:
    print(item)
print(len(similar_candidates))
'''
similar_items_list = []
for candidate_set in similar_candidates:
    potential_sim_items = list(combinations(candidate_set, 2))
    for couple in potential_sim_items:
        similarity = find_similarity(couple, rdd_by_business_id)
        if similarity >= 0.5:
            similar_items_list.append((couple[0], couple[1], similarity))

print(similar_items_list)'''


