from __future__ import absolute_import

import sys
import random
import logging
import unittest
import doctest
import collections
from itertools import islice
import numpy as np
from numpy import linalg as LA
from scipy.spatial import distance
import math
import random
import copy


file_string = "graph_representation_two_rounds_seed_0"
file_string_2 = "queries_two_rounds_seed_0"
num_queries = 700

num_dimensions = 100

L = 1
alpha = 2
R = 1

mediod = -100

# current_graph = {}

# already_seen = {}

# k_fixed = 6

# eps_fixed = 0.1

# failed_trials = 0

# num_seen = []
# approximation_ratio = []

def findL2Norm(vector):
    return LA.norm(vector)

def findL2Distance(vector1, vector2):
    return distance.euclidean(vector1, vector2)

def hyperbolicDistance(vector1, vector2):
    if (len(vector1) != num_dimensions):
        print("error vector1")
    if (len(vector2) != num_dimensions):
        print("error vector2")
    L2Distance = findL2Distance(vector1, vector2)

    vector1norm = findL2Norm(vector1)
    vector2norm = findL2Norm(vector2)

    inside = 1 + (2*np.square(L2Distance)) / ((1-np.square(vector1norm)) * (1- np.square(vector2norm)))

    return math.acosh(inside)

def getMinDistanceItem(query, list_of_items):
    #query is actually coordinates
    min_dist_so_far = 1000000
    p_star = list_of_items[0]
    num_in_list = len(list_of_items)
    for i in range(num_in_list):
        new_dist = hyperbolicDistance(query, list_of_items[i].coords)
        if new_dist < min_dist_so_far:
            min_dist_so_far = new_dist
            p_star = list_of_items[i]

    #returns an item
    return p_star



class Item(object):
    def __init__(self, embedding, word, dist, out_nodes, in_nodes):
        self.coords = embedding
        self.word = word
        self.distance_from_root = dist
        self.out_nodes = out_nodes
        self.in_nodes = in_nodes
    def __len__(self):
        return len(self.coords)
    def __getitem__(self, i):
        return self.coords[i]
    def __repr__(self):
        return 'Item({}, {}, {}, {}, {})'.format(self.coords, self.word, self.distance_from_root, self.out_nodes, self.in_nodes)
    def __hash__(self):
        return hash((self.coords.tobytes(), self.word, self.distance_from_root, self.out_nodes.tobytes(), self.in_nodes.tobytes()))

    def __eq__(self, other):
        return (self.coords.tobytes(), self.word, self.distance_from_root, self.out_nodes, self.in_nodes) == (other.coords.tobytes(), other.word, other.distance_from_root, other.out_nodes, other.in_nodes)



def loadGloveModel(gloveFile):
    #queryList = random.sample(104000, num_queries)
    pop = list(range(0, 104000))
    random.seed(0)
    queryList = random.sample(pop, num_queries)
    queryList.sort()
    #rand_index = 2086
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    points = {}
    points_array = []
    queries = []
    embedding_list = {}

    count_so_far = 0

    ##to determine mediod
    global mediod
    
    smallest_l2_norm = 10000
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        l2norm = findL2Norm(embedding)
        entity = Item(embedding, word, l2norm, [], [])
        if (l2norm > 0.) and (word != 'Scaling'):
            if (len(queryList) > 0) and (count_so_far == queryList[0]):
                if embedding.tobytes() not in embedding_list:
                    queries.append(entity)
                    queryList = queryList[1:]
                    embedding_list[embedding.tobytes()] = 0
            else:
                if embedding.tobytes() not in embedding_list:
                    points[word] = entity
                    points_array.append(entity)
                    if l2norm < smallest_l2_norm:
                        smallest_l2_norm = l2norm
                        mediod = entity
                    embedding_list[embedding.tobytes()] = 0
            count_so_far += 1

        # if count_so_far > num_data_points:
        #     break;
    print(count_so_far)

    print("Done.",len(points)," words loaded!")
    return points, points_array, queries

def initializeRandomGraph(initial_graph, points_array):
    num_nodes = len(initial_graph)
    for i in range(num_nodes):
        current_node_word = points_array[i].word
        initial_picked_out_nodes = np.random.choice(num_nodes-1, L, replace=False)
        for j in range(L):
            out_node_index = initial_picked_out_nodes[j]
            if out_node_index >= i:
                out_node_index += 1

            out_node_word = points_array[out_node_index].word

            initial_graph[current_node_word].out_nodes.append(out_node_word)
            initial_graph[out_node_word].in_nodes.append(current_node_word)
    return initial_graph


def get_ell_minus_vv(ell, vv):
    lst3 = [value for value in ell if value not in vv] 
    return lst3

def topLElts(query, list_of_items, num_to_take):
    #query is actually coordinates
    word_distance = {}
    for it in list_of_items:
        word_distance[it.word] = hyperbolicDistance(query, it.coords)

    sorted_dict = sorted(word_distance.items(), key=lambda x: x[1])
    num_dict_elts = len(sorted_dict)
    returnList = []
    count = 0
    for element in sorted_dict:
        returnList.append(element[0])
        count += 1
        if count == num_to_take:
            break

    return returnList

def unionTwoArraysOfItems(array1, array2):
    temp_dict={}
    for it in array1:
        temp_dict[it.word] = it
    for it2 in array2:
        temp_dict[it2.word] = it2

    union_array = []


    for ke in temp_dict.keys():
        union_array.append(temp_dict[ke])

    return union_array



def GreedySearch(initial_graph, s_item_obj, query_item, k, L_search):
    #ell and vv are arrays of Item objects
    ell = []
    s_item_word = s_item_obj.word
    ell.append(initial_graph[s_item_word])
    vv = []
    ell_minus_vv = get_ell_minus_vv(ell, vv)
    while len(ell_minus_vv) > 0:
        p_star = getMinDistanceItem(query_item.coords, ell_minus_vv)
        p_star_graph_node = initial_graph[p_star.word]
        vv.append(p_star_graph_node)

        out_node_word_array = initial_graph[p_star.word].out_nodes
        temp_ell = []
        for word in out_node_word_array:
            temp_ell.append(initial_graph[word])

        ell = unionTwoArraysOfItems(ell, temp_ell)
            
        num_ell_elts = len(ell)
        if num_ell_elts > L_search:
            truncatedList = topLElts(query_item.coords, ell, L_search)
            ell = []
            for word in truncatedList:
                ell.append(initial_graph[word])

        ell_minus_vv = get_ell_minus_vv(ell, vv)


    final_truncated_list = topLElts(query_item.coords, ell, k)
    ell = []
    for word in final_truncated_list:
        ell.append(initial_graph[word])


    return ell, vv


def robustPrune(initial_graph, query_item, vv, alpha, R):

    temp_weeded_vv = []

    out_node_words = initial_graph[query_item.word].out_nodes

    for word in out_node_words:
        if word != query_item.word:
            temp_weeded_vv.append(initial_graph[word])

    weeded_vv = []
    for i in range(len(vv)):
        if vv[i].word != query_item.word:
            weeded_vv.append(vv[i])

    weeded_vv = unionTwoArraysOfItems(temp_weeded_vv, weeded_vv)

    out_node_words = []

    while len(weeded_vv) > 0:
        p_star = getMinDistanceItem(query_item.coords, weeded_vv)
        out_node_words.append(p_star.word)
        if len(out_node_words) == R:
            break
        new_weeded_vv = []
        for j in range(len(weeded_vv)):
            if alpha*hyperbolicDistance(weeded_vv[j].coords, p_star.coords) > hyperbolicDistance(query_item.coords, weeded_vv[j].coords):
                new_weeded_vv.append(weeded_vv[j])

        weeded_vv = copy.deepcopy(new_weeded_vv)

    initial_graph[query_item.word].out_nodes = out_node_words

    return initial_graph


################## Actual Run #######################################

#points, points_array, queries = loadGloveModel('hypernym_noun.100d.txt')
points, points_array, queries = loadGloveModel('noun_embeddings_glove.txt')
print(points)
print(points_array)
print(queries)
print("finished loading")
print("original mediod is")
print(points_array[0])
print("new mediod is")
print(mediod)

graph = copy.deepcopy(points)

initialized_graph = initializeRandomGraph(graph, points_array)

random_permutation_1 = np.random.permutation(len(points_array))
random_permutation_2 = np.random.permutation(len(points_array))

#random_permutation = np.concatenate(random_permutation_1, random_permutation_2)

twice_points_array = 2*len(points_array)
for i in range(twice_points_array):
    if i < len(points_array):
        rand_index = random_permutation_1[i]
    else:
        rand_index = random_permutation_2[i - len(points_array)]
        
    print("Working on " + str(i) + " node")
    ell, vv = GreedySearch(initialized_graph, mediod, points_array[rand_index], 1, L)
    initialized_graph = robustPrune(initialized_graph, points_array[rand_index], vv, alpha, R)

    out_node_words = initialized_graph[points_array[rand_index].word].out_nodes

    for word in out_node_words:
        out_words_of_word = initialized_graph[word].out_nodes
        new_vv = [initialized_graph[points_array[rand_index].word]]
        for k in out_words_of_word:

            if k != points_array[rand_index].word:
                new_vv.append(initialized_graph[k])

        if len(new_vv) > R:
            initialized_graph = robustPrune(initialized_graph, initialized_graph[word], new_vv, alpha, R)
        else:
            initialized_graph[word].out_nodes = []
            for j in range(len(new_vv)):
                initialized_graph[word].out_nodes.append(new_vv[j].word)

file = open(file_string, "w")

for key in initialized_graph.keys():
    file.write(key)
    file.write(":")
    file.write(str(initialized_graph[key].out_nodes))
    file.write("\n")

file.close()

file = open(file_string_2, "w")
for q in queries:
    file.write(q.word)
    file.write(",")
file.close()






