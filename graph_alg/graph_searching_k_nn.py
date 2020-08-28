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
import heapq
import time

file_string = "test1"
graph_file_url = "../nouns_graph_seed0_L10_R10_alpha1-5"
queries_file_url = "../nouns_queries_seed0_L10_R10_alpha1"
glove_file = "../noun_embeddings_glove.txt"
#file_string_2 = "queries"
#num_queries = 1000

num_dimensions = 10

L = 10
alpha = 1
R = 10

global_count = 0
sampling_budget = 100
k_val = 1

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
	global global_count
	global seen_dict
	min_dist_so_far = 1000000
	p_star = list_of_items[0]
	num_in_list = len(list_of_items)
	for i in range(num_in_list):
		if (list_of_items[i].word not in seen_dict) and (global_count < sampling_budget):
			seen_dict[list_of_items[i].word] = 1
			global_count += 1

		new_dist = hyperbolicDistance(query, list_of_items[i].coords)
		if new_dist < min_dist_so_far:
			min_dist_so_far = new_dist
			p_star = list_of_items[i]

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



def get_ell_minus_vv(ell, vv):
	lst3 = [value for value in ell if value not in vv] 
	return lst3

def topLElts(query, list_of_items, num_to_take):
	global global_count
	global seen_dict
	#query is actually coordinates
	word_distance = {}
	for it in list_of_items:
		if it.word not in seen_dict:
			seen_dict[it.word] = 1
			global_count += 1
		word_distance[it.word] = hyperbolicDistance(query, it.coords)

	sorted_dict = sorted(word_distance.items(), key=lambda x: x[1])
	print(sorted_dict)
	num_dict_elts = len(sorted_dict)
	returnList = []
	count = 0
	for element in sorted_dict:
		returnList.append(element[0])
		count += 1
		if count == num_to_take:
			break

	return returnList

def topKDistances(query, list_of_items, num_to_take):
	top_k = []
	top_k_seen_dict = {}

	for it in list_of_items:
		if it.word not in top_k_seen_dict:
			top_k_seen_dict[it.word] = 1
			dis = hyperbolicDistance(query, it.coords)
			if len(top_k) >= num_to_take:
				if -dis > top_k[0]:
					heapq.heapreplace(top_k, -dis)
			else:
				heapq.heappush(top_k, -dis)
	return top_k

def findClosestGroundTruthPoint(query, dict_of_items, k_val_input):
	min_dist = 1000000
	current_best = ''
	top_k_ground_truth = []
	for w in dict_of_items.keys():
		temp_dist = hyperbolicDistance(query, dict_of_items[w].coords)
		if temp_dist < min_dist:
			min_dist = temp_dist
			current_best = w
		if len(top_k_ground_truth) >= k_val_input:
			if -temp_dist > top_k_ground_truth[0]:
				heapq.heapreplace(top_k_ground_truth, -temp_dist)
		else:
			heapq.heappush(top_k_ground_truth, -temp_dist)
	return current_best, min_dist, top_k_ground_truth

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
	#ell and vv are lists of Item objects
	ell = []
	s_item_word = s_item_obj.word
	ell.append(initial_graph[s_item_word])
	vv = []
	ell_minus_vv = get_ell_minus_vv(ell, vv)
	while len(ell_minus_vv) > 0 and global_count < sampling_budget:
		p_star = getMinDistanceItem(query_item.coords, ell_minus_vv)
		p_star_graph_node = initial_graph[p_star.word]
		vv.append(p_star_graph_node)

		out_node_word_array = initial_graph[p_star.word].out_nodes
		temp_ell = []
		print(p_star.word)
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





def loadGloveModel(gloveFile):
	#queryList = random.sample(104000, num_queries)
	# pop = list(range(0, 104000))
	# queryList = random.sample(pop, num_queries)
	# queryList.sort()
	#rand_index = 2086
	print("Loading Glove Model")
	f = open(gloveFile,'r')
	points = {}
	points_array = []
	# queries = []
	# embedding_list = {}

	for line in f:
		splitLine = line.split()
		word = splitLine[0]
		embedding = np.array([float(val) for val in splitLine[1:]])
		l2norm = findL2Norm(embedding)
		entity = Item(embedding, word, l2norm, [], [])
		if (l2norm > 0.) and (word != 'Scaling'):
			points[word] = entity
			points_array.append(entity)

		# if count_so_far > num_data_points:
		#     break;

	print("Done.",len(points)," words loaded!")
	return points, points_array



############## where all the real work is done #################


seen_dict = {}

############ working #####################
unfiltered_points, points_array = loadGloveModel(glove_file)

tossed_out_points = {}
points = {}

file = open(graph_file_url, "r")
for line in file:
	word_out_pair = line.split(':')
	#out_nodes_list = word_out_pair[1].split('\'')
	out_nodes_list = word_out_pair[1].split('\"')
	out_nodes_list_final = []
	for j in range(len(out_nodes_list)):
		if j%2 == 1:
			out_nodes_list_final.append(out_nodes_list[j])
		else:
			broken_list = out_nodes_list[j].split('\'')
			for k in range(len(broken_list)):
				if k%2 == 1:
					out_nodes_list_final.append(broken_list[k])
	unfiltered_points[word_out_pair[0]].out_nodes = out_nodes_list_final
	unfiltered_points[word_out_pair[0]].in_nodes = ['tag']

file.close()

for key in unfiltered_points.keys():
	if (unfiltered_points[key].out_nodes == [] and unfiltered_points[key].in_nodes == []):
		tossed_out_points[key] = unfiltered_points[key]
	else:
		points[key] = unfiltered_points[key]


############ working #####################

file2 = open(queries_file_url, "r")
queryLine = file2.readline()

queries = queryLine.split(",")
#queries = queries[:-1]

queries = queries[:3]
# print(queries[0])
# print(queries[1])
# print(queries[-1])
# print(len(queries))
file2.close()

##stats
num_invocations = []
final_hyperbolic_distance = []
true_hyperbolic_distance = []
failed_trials = 0
approximation_ratio = []
####

k_approximation_ratio = []
averaged_k_approximation_ratio = []
max_k_approximation_ratio = []
recall = []

time_elapsed_vec = []



len_dictionary = len(points)
initial_point = -1

num_queries = len(queries)

for qu in queries:
	global_count = 0

	found_in_each_round = []

	rand_init = random.sample(range(len(points)), 20)
	rand_init.sort()

	initial_point_array = []

	curr_count = 0
	curr_index = 0

	for nodes in points.keys():
		if curr_count == rand_init[curr_index]:
			initial_point_array.append(points[nodes])
			curr_index += 1
			if curr_index == 20:
				break
		curr_count +=1

	curr_index_init = 0

	start_time = time.time()

	while global_count < sampling_budget:
		initial_point = initial_point_array[curr_index_init]

		# randval = random.randint(0, len(points)-1)

		# curr_count = 0

		# for nodes in points.keys():
		# 	if curr_count == randval:
		# 		initial_point = points[nodes]
		# 		break
		# 	else:
		# 		curr_count += 1

		# print("how many points are there? ")
		# print(len(points))

		ell, vv = GreedySearch(points, initial_point, tossed_out_points[qu], k_val, L)
		found_in_each_round.append(ell)
		curr_index_init += 1

		if curr_index_init == 20:
			break

		# print(global_count)
		# print(ell)
	end_time = time.time()



	time_elapsed = end_time - start_time
	time_elapsed_vec.append(time_elapsed)

	query_embedding = tossed_out_points[qu].coords

	flattened = [val for sublist in found_in_each_round for val in sublist]
	top_k = topKDistances(query_embedding, flattened, k_val)
	# print("the distance we found is")
	# best_best_final = found_in_each_round[0]
	# best_best_final_distance = 100000
	# for j_item in found_in_each_round:
	# 	cand_dist = hyperbolicDistance(j_item.coords, query_embedding)
	# 	if cand_dist < best_best_final_distance:
	# 		best_best_final_distance = cand_dist
	# 		best_best_final = j_item

	num_invocations.append(global_count)
	#final_hyperbolic_distance.append(best_best_final_distance)

	ground_truth_nn_item, ground_truth_nn_dist, top_k_ground_truth = findClosestGroundTruthPoint(query_embedding, points, k_val)


	nptop_k = np.sort(np.asarray(top_k))
	nptop_k_ground_truth = np.sort(np.asarray(top_k_ground_truth))
	nptop_k_ground_truth = nptop_k_ground_truth[0:len(top_k)]

	approx_ratio_k_vec = nptop_k/nptop_k_ground_truth


	k_approximation_ratio.append(approx_ratio_k_vec)
	max_k_approximation_ratio.append(np.max(approx_ratio_k_vec))
	averaged_k_approximation_ratio.append(np.average(approx_ratio_k_vec))

	temp_recall = 0

	for a in range(len(top_k)):
		if approx_ratio_k_vec[a] <= 1.001:
			temp_recall += 1
	recall.append(float(temp_recall)/k_val)

	# true_hyperbolic_distance.append(ground_truth_nn_dist)
	# if best_best_final_distance > ground_truth_nn_dist:
	# 	failed_trials += 1
	# 	approximation_ratio.append(float(best_best_final_distance) / float(ground_truth_nn_dist))


total_invocations = sum(num_invocations)
average_invocations = float(total_invocations) / float(num_queries)

max_invocations = max(num_invocations)
min_invocations = min(num_invocations)
invocations_sd = np.std(num_invocations)

avg_max_k_approximation_ratio = np.average(max_k_approximation_ratio)
avg_average_k_approximation_ratio = np.average(averaged_k_approximation_ratio)

avg_recall = np.average(recall)

avg_time_elapsed = np.average(time_elapsed_vec)

avg_approx = -1
max_approx = -1
min_approx = -1
approx_sd = -1

if failed_trials > 0:
	total_approx = sum(approximation_ratio)
	avg_approx = float(total_approx) / float(failed_trials)
	max_approx = max(approximation_ratio)
	min_approx = min(approximation_ratio)

	approx_sd = np.std(approximation_ratio)


file = open(file_string, "w")

file.write("number of queries is: ")
file.write("\n")
file.write(str(num_queries))
file.write("\n")
file.write("\n")

file.write("number of elements in the dataset is: ")
file.write("\n")
file.write(str(len_dictionary))
file.write("\n")
file.write("\n")

file.write("average invocations is: ")
file.write("\n")
file.write(str(average_invocations))
file.write("\n")
file.write("\n")

file.write("max invocations is: ")
file.write("\n")
file.write(str(max_invocations))
file.write("\n")
file.write("\n")

file.write("min invocations is: ")
file.write("\n")
file.write(str(min_invocations))
file.write("\n")
file.write("\n")

file.write("standard deviation invocations is: ")
file.write("\n")
file.write(str(invocations_sd))
file.write("\n")
file.write("\n")


file.write("number of failed trials is: ")
file.write("\n")
file.write(str(failed_trials))
file.write("\n")
file.write("\n")


file.write("approximation ratio matrix is: ")
file.write("\n")
file.write(str(approximation_ratio))
file.write("\n")
file.write("\n")


file.write("average approx is: ")
file.write("\n")
file.write(str(avg_approx))
file.write("\n")
file.write("\n")

file.write("max approx is: ")
file.write("\n")
file.write(str(max_approx))
file.write("\n")
file.write("\n")

file.write("min approx is: ")
file.write("\n")
file.write(str(min_approx))
file.write("\n")
file.write("\n")

file.write("standard deviation approx is: ")
file.write("\n")
file.write(str(approx_sd))
file.write("\n")
file.write("\n")

file.write("averaged max approx ratio: ")
file.write("\n")
file.write(str(avg_max_k_approximation_ratio))
file.write("\n")
file.write("\n")

file.write("averaged average k approx ratio: ")
file.write("\n")
file.write(str(avg_average_k_approximation_ratio))
file.write("\n")
file.write("\n")

file.write("average k approx ratio per query: ")
file.write("\n")
file.write(str(averaged_k_approximation_ratio))
file.write("\n")
file.write("\n")

file.write("max k approx ratio per query: ")
file.write("\n")
file.write(str(max_k_approximation_ratio))
file.write("\n")
file.write("\n")

file.write("approx ratio per query: ")
file.write("\n")
file.write(str(k_approximation_ratio))
file.write("\n")
file.write("\n")

file.write("approx recall per query: ")
file.write("\n")
file.write(str(avg_recall))
file.write("\n")
file.write("\n")

file.write("average time elapsed: ")
file.write("\n")
file.write(str(avg_time_elapsed))
file.write("\n")
file.write("\n")

file.close()







