import os

import pandas as pd
from sklearn.model_selection import train_test_split
import random

'''
KGRec - music dataset = https://www.upf.edu/web/mtg/kgrec 

By using 'implicit_lf_dataset.csv' it creates 5 random splits and train validation and test sets
'''
WORKING_DIR = os.getcwd()
MAIN_DIR = os.path.abspath(os.path.join(WORKING_DIR, os.pardir))
DATA_DIR = os.path.join(MAIN_DIR, "data/kgrec/")
OUT_DIR = DATA_DIR


def get_count(tp, id):
    playcount_groupbyid = tp[[id, 'rating']].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


def print_stats(tp):
    usercount, moviecount = get_count(tp, 'userId'), get_count(tp, 'itemId')

    sparsity_level = float(tp.shape[0]) / (usercount.shape[0] * moviecount.shape[0])
    print("There are %d triplets from %d users and %d songs (sparsity level %.3f%%)" % (tp.shape[0],
                                                                                        usercount.shape[
                                                                                            0],
                                                                                        moviecount.shape[
                                                                                            0],
                                                                                        sparsity_level * 100))


def numerize(tp, user2id, movie2id):
    uid = list(map(lambda x: user2id[x], tp['userId']))
    sid = list(map(lambda x: movie2id[x], tp['itemId']))
    #print(uid[1])
    tp['userId'] = uid
    tp['itemId'] = sid
    return tp


def filter_triplets(tp, min_uc=20, min_sc=1):
    # Only keep the triplets for songs which were listened to by at least min_sc users.
    moviecount = get_count(tp, 'itemId')
    tp = tp[tp['itemId'].isin(moviecount.index[moviecount >= min_sc])]

    # Only keep the triplets for users who listened to at least min_uc songs
    # After doing this, some of the songs will have less than min_uc users, but should only be a small proportion
    usercount = get_count(tp, 'userId')
    tp = tp[tp['userId'].isin(usercount.index[usercount >= min_uc])]
    return tp


raw_data = pd.read_csv(os.path.join(DATA_DIR, 'implicit_lf_dataset.csv'), header=0, sep='\t')
print_stats(raw_data)

raw_data = filter_triplets(raw_data, min_uc=20, min_sc=1)
print_stats(raw_data)
# Map the string ids to unique incremental integer ids for both users and songs
usercount, songcount = get_count(raw_data, 'userId'), get_count(raw_data, 'itemId')
unique_uid = usercount.index
unique_sid = songcount.index
song2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))

with open(os.path.join(OUT_DIR, 'users.txt'), 'w') as f:
    for uid in unique_uid:
        f.write('%s\n' % user2id[uid])
f.close()
with open(os.path.join(OUT_DIR, 'items.txt'), 'w') as f:
    for sid in unique_sid:
        f.write('%s\n' % song2id[sid])
f.close()

random.seed(2812020)
#  Create train/validation/test sets, five different random splits
for i in range(1, 6):
    FOLD_DIR = os.path.join(OUT_DIR, str(i))
    if not os.path.exists(FOLD_DIR):
        os.makedirs(FOLD_DIR)
    seed = random.randint(0, 1000000)
    print("seed : " + str(seed))
    train_validation, test = train_test_split(raw_data, test_size=0.2, stratify=raw_data.userId, shuffle=True,
                                              random_state=seed)
    train, validation = train_test_split(train_validation, test_size=0.25, stratify=train_validation.userId,
                                         shuffle=True,
                                         random_state=seed)

    tv_tp = numerize(train_validation, user2id, song2id)
    tv_tp.to_csv(os.path.join(FOLD_DIR, 'train.csv'), index=False, header=False)

    train_tp = numerize(train, user2id, song2id)
    train_tp.to_csv(os.path.join(FOLD_DIR, 't.csv'), index=False, header=False)

    test_tp = numerize(test, user2id, song2id)
    test_tp.to_csv(os.path.join(FOLD_DIR, 'test.csv'), index=False, header=False)

    vad_tp = numerize(validation, user2id, song2id)
    vad_tp.to_csv(os.path.join(FOLD_DIR, 'val.csv'), index=False, header=False)

# Since we mapped the IDs, save the corresponding ratings.
raw_data = numerize(raw_data, user2id, song2id)
raw_data.to_csv(os.path.join(DATA_DIR, 'ratings.csv'), index=False, header=False)