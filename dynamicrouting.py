import sys
from ortools.sat.python import cp_model
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math
from random import randint, random, sample
from time import sleep, time
import csv
import codecs
import json
from classes import Intersection, Endpoint, Node, Outage, Car, Route, Routes
from rtree import Node
from dynamicdistances import get_distance_matrix
from distances import euclidean_distance
from weather_learning import predict_fix_times
import datetime
from subprocess import Popen
from statistics import median, stdev, mean


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

def get_distance_to_outages(routes, data, outage_index):
  new_routes = []
  for route in routes:
    for i in range(len(route.paths)):
      new_paths = []
      total_time = 0.0
      for j in range(len(route.paths)):
        if i == j:
          new_paths.append(Route(route.paths[i].vehicle_id, outage_index, route.paths[i].previous_indices + [outage_index], route.paths[i].route_time + data[route.paths[i].current_index][outage_index]))
          total_time += data[route.paths[i].current_index][outage_index]
        else:
          new_paths.append(route.paths[j])
        total_time += route.paths[j].route_time
      new_routes.append(Routes(new_paths, total_time))
  return new_routes
  

def insertion_based_routing(data, outages, prune_constant):
  start = time()
  def get_time(elem):
    return elem.max_time

  starting_vehicles = [Route(index, len(outages) + 1 + index, [len(outages) + 1 + index], 0.0) for index in range(len(data) - len(outages) - 1)]
  routes = [Routes(starting_vehicles, 0.0)]
  for outage_index in range(1, len(outages) + 1):
    routes = get_distance_to_outages(routes, data, outage_index)
    routes.sort(key=get_time)
    routes = routes[:prune_constant * (outage_index + 1)]
  
  min_route = min(routes, key=get_time)
  end = time()
  difference = end - start
  return print_info(min_route, difference)

def get_rtree_routes(routes, outage_index, outage, cars_to_check, prune_constant):
  def get_distance(elem):
    return elem[1]

  distances = []
  for car in cars_to_check:
    distances.append((car.id, math.sqrt((car.latitude - outage.latitude) ** 2 + (car.longitude - outage.longitude) ** 2) + 0.1))
  distances.sort(key=get_distance)
  distances = distances[:prune_constant]

  new_routes = []
  for route in routes:
    for edge in distances:
      new_paths = []
      total_time = 0.0
      for i in range(len(route.paths)):
        if route.paths[i].vehicle_id == edge[0]:
          new_paths.append(Route(route.paths[i].vehicle_id, outage_index, route.paths[i].previous_indices + [outage_index], route.paths[i].route_time + edge[1]))
          total_time += edge[1]
        else:
          new_paths.append(route.paths[i])
        total_time += route.paths[i].route_time
      new_routes.append(Routes(new_paths, total_time))
  return new_routes

def rtree_based_routing(data, outages, prune_constant, vehicles):
  start = time()
  def get_time(elem):
    return elem.max_time

  starting_vehicles = [Route(index, len(outages) + 1 + index, [len(outages) + 1 + index], 0.0) for index in range(len(data) - len(outages) - 1)]
  routes = [Routes(starting_vehicles, 0.0)]
  rtree = Node(True, 0)
  for vehicle in vehicles:
    rtree.insert_new_value(vehicle, prune_constant)
  for outage_index in range(len(outages)):
    node = rtree.insert_new_value(outages[outage_index], prune_constant)
    values = rtree.return_all_values(node)
    cars_to_check = [value for value in values if value.is_car]
    routes = get_rtree_routes(routes, outage_index, outages[outage_index], cars_to_check, prune_constant)
    routes.sort(key=get_time)
    routes = routes[:prune_constant * (outage_index + 1)]
  
  min_route = min(routes, key=get_time)
  end = time()
  difference = end - start
  return print_info(min_route, difference)

def print_info(route, time_difference):
  times = []
  outages = []

  for path in route.paths:
    times.append(path.route_time)
    outages.append(len(path.previous_indices) - 1)

  average_time = mean(times)
  median_time = median(times)
  min_time = min(times)
  max_time = max(times)
  standard_deviation_time = stdev(times)

  average_outages = mean(outages)
  median_outages = median(outages)
  min_outages = min(outages)
  max_outages = max(outages)
  standard_deviation_outages = stdev(outages)

  return [time_difference, average_time, median_time, min_time, max_time, standard_deviation_time,
    average_outages, median_outages, min_outages, max_outages, standard_deviation_outages]


def get_outages():
  f = open('./outages1.json', 'r')
  return json.load(f)

def create_recursive_rtree(rtree, index):
  if rtree.is_leaf:
    values_dict = { }
    for i in range(len(rtree.values)):
      values_dict[i] = rtree.values[i].convert_to_dict()
    rtree.values = values_dict
    return { index: rtree.convert_to_dict() }
  return { index: rtree.convert_to_dict(), "left": create_recursive_rtree(rtree.nodes[0], index + 1), "right": create_recursive_rtree(rtree.nodes[1], index + 1) }

def assemble_rtree(outages):
  rtree = Node(True, 0)
  for outage in outages:
    rtree.insert_new_value(outage)
  return rtree

def main(number_of_cars, outages=None, type_of_prediction="actual", prune_constant=1):
  if outages == None:
    outages = get_outages()
  intersection_dictionary = get_routing_data()
  prediction_times = predict_fix_times(type_of_prediction, outages)
  outages = define_outages(intersection_dictionary, outages, prediction_times)

  return_object = { }
  for i in range(5):
    starting_intersections = sample(list(intersection_dictionary.values()), number_of_cars)
    vehicles = [Car(index, starting_intersections[index].latitude, starting_intersections[index].longitude, starting_intersections[index]) for index in range(len(starting_intersections))]
    distance_matrix = get_distance_matrix([None] + outages + vehicles, intersection_dictionary)
    insertion_return_object = insertion_based_routing(distance_matrix, outages, prune_constant)
    rtree_return_object = rtree_based_routing(distance_matrix, outages, prune_constant, vehicles)
    return_object[i] = insertion_return_object + rtree_return_object
  
  #json_object = json.dumps(create_recursive_rtree(rtree, 0))
  print(json.dumps(return_object))

main(int(sys.argv[1]), json.loads(sys.argv[2]), str(sys.argv[3]), int(sys.argv[4]))