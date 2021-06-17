import sys
from ortools.sat.python import cp_model
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math
from random import randint, random, sample
from time import sleep
import csv
import codecs
import json
from classes import Intersection, Endpoint, Node, Outage, Car
from kmeans import k_means
from kmeansplusplus import k_means_plus_plus
from distances import closest_centroid, get_distance_matrix, euclidean_distance
from weather_learning import predict_fix_times
import datetime
from subprocess import Popen
from static_case import vehicular_routing_problem_static_case, create_solution_static_case
from base_case import vehicular_routing_problem_base_case


# intersections.csv contains the necessary intersection data
# USA-road contains the TIMES between the intersections
def get_routing_data():
  intersection_dictionary = { }
  f = open('./routes/pathfinding/intersections.csv', 'r')
  reader = csv.reader(f, delimiter=',')
  for row in reader:
    intersection_dictionary[int(row[1])] = Intersection(row[0], int(row[1]), float(row[2]), float(row[3]))
  f = open('./routes/pathfinding/USA-road-t.NY.gr', 'r')
  for row in f:
    words = row.split(' ')
    if len(words) == 4 and words[0] == 'a' and int(words[1]) in intersection_dictionary and int(words[2]) in intersection_dictionary:
      intersection_dictionary[int(words[1])].endpoints[int(words[2])] = Endpoint(int(words[2]), int(words[3]))
  return intersection_dictionary

def define_outages(intersection_dictionary, outages, prediction_times):
  outage_objects = []
  i = 0
  for row in outages:
    outage_objects.append(Outage(i, float(row['Latitude']), float(row['Longitude']), intersection_dictionary[7440], row['TimeStampCrawled'], prediction_times[i][0], prediction_times[i][1]))
    i += 1
  i = 0
  for outage in outage_objects:
    for intersection in intersection_dictionary.values():
      current_outage_distance = euclidean_distance(outage.latitude, outage.longitude, outage.closest_intersection.latitude, outage.closest_intersection.longitude)
      potential_outage_distance = euclidean_distance(outage.latitude, outage.longitude, intersection.latitude, intersection.longitude)
      if potential_outage_distance < current_outage_distance:
        outage.closest_intersection = intersection
    i += 1
  return outage_objects

def main(algorithm, number_of_outages, number_of_cars, number_of_centroids, outages, type_of_prediction, type_of_test, type_of_matrix):
  intersection_dictionary = get_routing_data()
  prediction_times = predict_fix_times(type_of_prediction, outages)
  outages = define_outages(intersection_dictionary, outages, prediction_times)
  number_of_k_means_iterations = 100
  starting_intersections = sample(list(intersection_dictionary.values()), number_of_cars)
  cars = [Car(index, starting_intersections[index].latitude, starting_intersections[index].longitude, starting_intersections[index]) for index in range(len(starting_intersections))]
  clusters = None
  if algorithm == "kmeans":
    clusters = k_means(outages, number_of_centroids, number_of_k_means_iterations)
  elif algorithm == "kmeansplusplus":
    clusters = k_means_plus_plus(outages, number_of_centroids, number_of_k_means_iterations)
  centroid_keys = [index for index in range(number_of_centroids)]
  keys_and_cars = []
  i = 0
  while len(centroid_keys) > 0 and i < 200:
    centroid_keys = [index for index in range(number_of_centroids)]
    keys_and_cars = []
    for car in cars:
      (key, _, _, _) = closest_centroid(car.latitude, car.longitude, [cluster.centroid for cluster in clusters.values()])
      keys_and_cars.append([key, car])
    for key_and_car in keys_and_cars:
      index = -1
      try:
        index = centroid_keys.index(key_and_car[0])
      except ValueError:
        index = -1
      if index != -1:
        centroid_keys.pop(index)
    i += 1
  for key_and_car in keys_and_cars:
    clusters[key_and_car[0]].vehicles.append(key_and_car[1])
  all_data = []
  for cluster in clusters.values():
    data = { }
    info = None
    routing_matrix = None
    (data['distance_matrix'], routing_matrix) = get_distance_matrix([None] + cluster.outages + cluster.vehicles, intersection_dictionary, type_of_matrix)
    if len(data['distance_matrix'][0]) > 1 and len(cluster.vehicles) > 0:
      data['num_vehicles'] = len(cluster.vehicles)
      data['starts'] = [len(cluster.outages) + index + 1 for index in range(len(cluster.vehicles))] # need to add one because of dummy depot ends!!!
      data['ends'] = [0 for index in range(len(cluster.vehicles))]
      if type_of_test == "basecase":
        info = vehicular_routing_problem_base_case(data['distance_matrix'], cluster.outages)
      elif type_of_test == "staticcase":
        (data, manager, routing, solution) = vehicular_routing_problem_static_case(data)
        if solution:
          info = create_solution_static_case(data, manager, routing, solution)
    all_data.append([cluster, routing_matrix, info])
  json_object = { }
  i = 0
  for data_point in all_data:
    current_data_point = { }
    current_data_point["cluster"] = data_point[0].convert_to_dict()
    #current_data_point["routing_matrix"] = data_point[1]
    if data_point[2] != None:
      current_data_point["info"] = [piece.convert_to_dict() for piece in data_point[2]]
    json_object[i] = current_data_point
    i += 1
  json_object = json.dumps(json_object)
  print(json_object)

main(str(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), json.loads(sys.argv[5]), str(sys.argv[6]), str(sys.argv[7]), str(sys.argv[8]))