import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
import os
import sys

# Give a random seed for reproducibility
random.seed(111119)
NUMBER_OF_GROUPS_PER_CONF = 1000

'''
This code is used to create synthetic groups based on three different criteria:
1) Random groups: Users are randomly assigned to groups.
2) Similar groups: PCC between user pairs is computed. Then groups are constructed based on similarities. 
3) Divergent groups: PCC between user pairs is computed. Then groups are constructed based on similarities. 
'''


def load_user_ids(file_path):
    '''
    Load user ids from file.
    :param file_path:
    :return:
    '''
    users = pd.read_csv(file_path, header=None)
    return set(users[0].values)


def compute_pearson_correlation_coefficient(ratings_file_path, sep=',', user_column='user_id', item_column='item_id',
                                            rating_column='rating'):
    '''
    Computes pearson correlation coefficient between users in the dataset! Returns a numpy matrix for user-user
    similarity.
    :param ratings_file_path:
    :return:
    '''
    df = pd.read_csv(ratings_file_path, header=None, sep=sep,
                     names=[user_column, item_column, rating_column])
    user_matrix = df.pivot_table(columns=item_column, index=user_column, values=rating_column)

    print('pivot table!')
    user_id_indexes = user_matrix.index.values
    # Fill non-existing ratings with zeros!
    user_matrix = user_matrix.fillna(0)
    numpy_array = user_matrix.to_numpy()
    # Compute user-user similarity!
    sim_matrix = np.corrcoef(numpy_array)
    # sim_matrix = np.corrcoef(user_matrix.to_dense())

    return sim_matrix, user_id_indexes


def pcc_test(ratings_file_path, sep=',', user_column='user_id', item_column='item_id',
             rating_column='rating'):
    df = pd.read_csv(ratings_file_path, header=None, sep=sep,
                     names=[user_column, item_column, rating_column, 'timestamp'])
    user_matrix = df.pivot_table(columns=item_column, index=user_column, values=rating_column).dropna().to_sparse()
    sim_matrix = np.corrcoef(user_matrix)
    print('done')


def generate_random_groups(users, DATA_PATH):
    # Randomly generate groups of different sizes m = 2,3,4,5,6,7,8
    for group_size in range(2, 9):
        group_id = 0
        random_groups_file_name = "random_group_" + str(group_size)
        random_groups_file_path = os.path.join(DATA_PATH, random_groups_file_name)
        random_groups_file = open(random_groups_file_path, 'w')
        selection = random.sample(users, min(NUMBER_OF_GROUPS_PER_CONF, len(users)))
        for user in selection:
            group = random.sample(users.difference({user}), group_size - 1)
            group.append(user)
            group_str = '\t'.join(map(str, group))
            random_groups_file.write(str(group_id) + '\t' + group_str + '\n')
            group_id += 1
        random_groups_file.flush()
        random_groups_file.close()


def generate_similar_user_groups(ratings_file_path, DATA_PATH, users, sep, user_column, item_column, rating_column):
    '''
    Generate synthetic groups based on the PCC values between users of different group size m=2,3,...,8
    :param ratings_file_path:
    :param users:
    :return:
    '''
    sim_matrix, user_id_indexes = compute_pearson_correlation_coefficient(ratings_file_path, sep, user_column,
                                                                          item_column, rating_column)
    for group_size in range(2, 9):
        group_id = 0
        sim_groups_file_name = "sim_group_" + str(group_size)
        sim_groups_file_path = os.path.join(DATA_PATH, sim_groups_file_name)
        sim_groups_file = open(sim_groups_file_path, 'w')
        selection = random.sample(users, min(NUMBER_OF_GROUPS_PER_CONF, len(users)))
        for user in selection:
            group = [user]
            # Write a method that randomly selects a user that has a PCC >= 0.5 to the existing group members.
            for i in range(group_size - 1):
                selection = select_user_for_sim_group(group, sim_matrix, user_id_indexes)
                if selection is None:
                    break
                else:
                    group.extend(selection)
            if len(group) != group_size:
                continue
            group_str = '\t'.join(map(str, group))
            sim_groups_file.write(str(group_id) + '\t' + group_str + '\n')
            group_id += 1
        sim_groups_file.flush()
        sim_groups_file.close()


def generate_divergent_user_groups(ratings_file_path, DATA_PATH, users, sep, user_column, item_column, rating_column):
    '''
    Generate synthetic groups based on the PCC values between users of different group size m=2,3,...,8
    :param ratings_file_path:
    :param users:
    :return:
    '''
    sim_matrix, user_id_indexes = compute_pearson_correlation_coefficient(ratings_file_path, sep, user_column,
                                                                          item_column, rating_column)
    for group_size in range(2, 9):
        group_id = 0
        sim_groups_file_name = "div_group_" + str(group_size)
        sim_groups_file_path = os.path.join(DATA_PATH, sim_groups_file_name)
        sim_groups_file = open(sim_groups_file_path, 'w')
        selection = random.sample(users, min(NUMBER_OF_GROUPS_PER_CONF, len(users)))
        for user in selection:
            group = [user]
            # Write a method that randomly selects a user that has a PCC < 0.1 to the existing group members.
            for i in range(group_size - 1):
                selection = select_user_for_divergent_group(group, sim_matrix, user_id_indexes)
                if selection is None:
                    break
                else:
                    group.extend(selection)
            if len(group) != group_size:
                continue
            group_str = '\t'.join(map(str, group))
            sim_groups_file.write(str(group_id) + '\t' + group_str + '\n')
            group_id += 1
        sim_groups_file.flush()
        sim_groups_file.close()


def extract_pcc_stats(ratings_file_path):
    sim_matrix, user_id_indexes = compute_pearson_correlation_coefficient(ratings_file_path)
    print(len(sim_matrix[sim_matrix >= 0.8]))
    stats = dict()
    vals = [x for x in np.arange(-1, 1.1, 0.1)]
    for i in range(len(vals)):
        if i == (len(vals) - 1):
            break
        stats[vals[i]] = len(sim_matrix[np.logical_and(vals[i] <= sim_matrix, sim_matrix < vals[i + 1])]) / 2
    print(stats)
    plt.bar(list(stats.keys()), stats.values(), color='g')
    # plt.hist(stats)
    plt.show()


def select_user_for_sim_group(group, sim_matrix, user_id_indexes, sim_threshold=0.3):
    '''
    Helper function to the generate_similar_user_groups function. Given already selected group members, it randomly
    selects from the remaining users that has a PCC value >= sim_threshold to any of the existing members.
    :param group:
    :param sim_matrix:
    :param user_id_indexes:
    :param sim_threshold: 0.5 is large size effect, 0.3 medium and 0.1 is small based on research (add citation!)
    :return:
    '''
    ids_to_select_from = set()
    for member in group:
        member_index = user_id_indexes.tolist().index(member)
        indexes = np.where(sim_matrix[member_index] >= sim_threshold)[0].tolist()
        user_ids = [user_id_indexes[index] for index in indexes]
        ids_to_select_from = ids_to_select_from.union(set(user_ids))
    candidate_ids = ids_to_select_from.difference(set(group))
    if len(candidate_ids) == 0:
        return None
    else:
        selection = random.sample(candidate_ids, 1)
        return selection


def select_user_for_divergent_group(group, sim_matrix, user_id_indexes, sim_threshold=0.1):
    '''
    Helper function to the generate_similar_user_groups function. Given already selected group members, it randomly
    selects from the remaining users that has a PCC value < sim_threshold to any of the existing members.
    :param group:
    :param sim_matrix:
    :param user_id_indexes:
    :param sim_threshold: 0.5 is large size effect, 0.3 medium and 0.1 is small based on research (add citation!)
    :return:
    '''
    ids_to_select_from = set()
    for member in group:
        member_index = user_id_indexes.tolist().index(member)
        indexes = np.where(sim_matrix[member_index] < sim_threshold)[0].tolist()
        user_ids = [user_id_indexes[index] for index in indexes]
        ids_to_select_from = ids_to_select_from.union(set(user_ids))
    candidate_ids = ids_to_select_from.difference(set(group))
    if len(candidate_ids) == 0:
        return None
    else:
        selection = random.sample(candidate_ids, 1)
        return selection


def extract_group_pcc_stats(ratings_file_path, DATA_PATH, group_type, out_file_path, sep=',', user_column='userId',
                            item_column='itemId', rating_column='rating'):
    '''
    To plot box plots for the average pcc for groups, use this method.
    :param ratings_file_path:
    :param DATA_PATH:
    :param group_type:
    :param out_file_path:
    :return:
    '''
    sim_matrix, user_id_indexes = compute_pearson_correlation_coefficient(ratings_file_path, sep, user_column,
                                                                          item_column, rating_column)
    user_id_indexes_list = user_id_indexes.tolist()
    out_file = open(out_file_path, 'a')
    for group_size in range(2, 9):
        file_path = DATA_PATH + '/' + group_type + '_group_' + str(group_size)
        group_file = open(file_path, 'r')
        for line in group_file:
            line = line.split('\t')
            group_id = line[0]
            members = [int(line[i]) for i in range(1, len(line))]
            members_indexes = [user_id_indexes_list.index(member) for member in members]
            pcc_sum = 0.0
            for u in members_indexes:
                for v in members_indexes:
                    if u == v:
                        continue
                    pcc_sum += sim_matrix[u][v]
            avg_group_pcc = pcc_sum / (float)(len(members) * (len(members) - 1))
            out_file.write(group_type + ',' + group_id + ',' + str(avg_group_pcc) + '\n')
        out_file.flush()
    out_file.close()


def get_userIDs_in_groups(DATA_PATH, out_file_name):
    out_file = open(os.path.join(DATA_PATH, out_file_name), 'w')
    group_sizes = [i for i in range(2, 9)]
    ids = set()
    for group_size in group_sizes:
        input_file = open(os.path.join(DATA_PATH, 'random_group_') + str(group_size), 'r')
        for line in input_file:
            line = line.split('\t')
            for j in range(1, group_size + 1):
                ids.add(line[j])
        input_file.close()
    for id in ids:
        out_file.write(id + '\n')
    out_file.flush()
    out_file.close()


if __name__ == '__main__':
	DATASET = sys.argv[1]
    # ML1M dataset
	if DATASET == 'ML1M':
	    WORKING_DIR = os.getcwd()
	    MAIN_DIR = os.path.abspath(os.path.join(WORKING_DIR, os.pardir))
	    DATA_DIR = os.path.join(MAIN_DIR, "data/ml1m/")
	    users_file_path = os.path.join(DATA_DIR, "users.txt")
	    ratings_file_path = os.path.join(DATA_DIR, "ratings.csv")
	    users = set(load_user_ids(users_file_path))
	    generate_random_groups(users, DATA_DIR)
	    # Create groups of similar users!
	    # First, Compute pearson correlation coefficient between users
	    generate_similar_user_groups(ratings_file_path, DATA_DIR, users, ',', 'userId', 'itemId', 'rating')
	    generate_divergent_user_groups(ratings_file_path, DATA_DIR, users, ',', 'userId', 'itemId', 'rating')

	    # For extracting statistics for the groups, Output of this can be used by R code to generate Figure 1 of the paper!
	    '''for group_type in ["random", "sim", "div"]:
	        extract_group_pcc_stats(ratings_file_path, DATA_DIR, group_type, os.path.join(DATA_DIR, "random_pcc.txt"))'''

	elif DATASET == 'KGREC':
    # KGRec-Music dataset
	    WORKING_DIR = os.getcwd()
	    MAIN_DIR = os.path.abspath(os.path.join(WORKING_DIR, os.pardir))
	    DATA_DIR = os.path.join(MAIN_DIR, "data/kgrec/")
	    users_file_path = os.path.join(DATA_DIR, "users.txt")
	    ratings_file_path = os.path.join(DATA_DIR, "ratings.csv")
	    users = set(load_user_ids(users_file_path))
	    generate_random_groups(users, DATA_DIR)
	    # TODO replace movieId below with songId
	    generate_similar_user_groups(ratings_file_path, DATA_DIR, users, ',', 'userId', 'itemId', 'rating')
	    generate_divergent_user_groups(ratings_file_path, DATA_DIR, users, ',', 'userId', 'itemId', 'rating')

	    # For extracting statistics for the groups, Output of this can be used by R code to generate Figure 1 of the paper!
	    '''for type in ["random", "sim", "div"]:
	        extract_group_pcc_stats(ratings_file_path, DATA_DIR, type, os.path.join(DATA_DIR, "random_pcc.txt"))'''
