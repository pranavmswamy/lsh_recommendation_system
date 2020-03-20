from pyspark import SparkContext
from itertools import combinations
from sys import argv

sc = SparkContext()


# -----------------------------
def hash_to_32_bit_nums(businesses_itr):
    businesses = list(businesses_itr)
    return [hash(business_id) % 4294967295 for business_id in businesses]


def h1(x):
    a = 2976840907
    b = 3727260824
    return (a * x + b)


def h2(x):
    a = 2100863184
    b = 634485346
    return (a * x + b)


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

    r = 2
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
        band_row_hash = h1(int(combined_str)) % (c)
        band_hash_list.append(((band_row_hash), sig_key_column[0]))

    return band_hash_list


def find_similarity(couple):
    shingles1 = set(users_by_business_id[couple[0]])
    shingles2 = set(users_by_business_id[couple[1]])
    similarity = len(shingles1.intersection(shingles2)) / len(shingles1.union(shingles2))
    if similarity >= 0.5:
        return tuple(sorted((couple[0], couple[1]))), similarity
    else:
        return '<0.5'


# -----------------------------

rdd = sc.textFile(str(argv[1])).filter(lambda s: s.startswith('user_id') is False) \
    .map(lambda s: tuple(s.split(",")[-2::-1]))  # business_id, user_id

rdd_by_business_id = rdd.groupByKey().persist()

users_by_business_id = dict(rdd_by_business_id.collect())

rdd_by_business = rdd_by_business_id.mapValues(hash_to_32_bit_nums)

sig_matrix_rdd = rdd_by_business.mapValues(convert_to_sig_matrix)

# checked upto here, working properly - sig matrix is correct. Tested with two businesses on Piazza, got a val of 0.65 (Piazza val - 0.66).

lsh_rdd = sig_matrix_rdd.map(lsh)
lsh_rdd = lsh_rdd.flatMap(lambda l: l).groupByKey().mapValues(lambda l: list(l))
lsh_rdd = lsh_rdd.filter(lambda x: len(x[1]) > 1)

sim_items_candidates_rdd = lsh_rdd.map(lambda x: tuple(sorted(x[1]))).distinct()

sim_items_candidates_rdd = sim_items_candidates_rdd.flatMap(lambda items: combinations(items, 2)).map(
    lambda couple: tuple(sorted(couple))).distinct()

sim_items_candidates_rdd = sim_items_candidates_rdd.map(find_similarity).filter(lambda x: x != '<0.5')

similar_candidates = sim_items_candidates_rdd.collectAsMap()

for item, rating in similar_candidates.items():
    print(item, rating)

print(len(similar_candidates))


# writing to file
with open(str(argv[2]), "w") as file:
    file.write("business_id_1, business_id_2, similarity")
    for business_ids, rating in sorted(similar_candidates.items(), key=lambda x: (x[0][0], x[0][1])):
        file.write(str("\n" + business_ids[0] + "," + business_ids[1] + "," + str(rating)))
    file.close()


# Ground Truth
ground_truth = sc.textFile("pure_jaccard_similarity.csv").map(lambda data: data.split(",")).filter(
    lambda data: data[0] != "business_id_1").map(lambda data: ((data[0], data[1]), data[2])).collectAsMap()
true_positive = 0
false_positive = 0
false_negative = 0
for ids in similar_candidates:
    if ids in ground_truth:
        true_positive += 1
    else:
        false_positive += 1
for ids in ground_truth:
    if ids not in similar_candidates:
        false_negative += 1

print("Precision:", true_positive / (true_positive + false_positive))
print("Recall:", true_positive / (true_positive + false_negative))