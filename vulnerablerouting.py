import sys
from ortools.sat.python import cp_model
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math
from random import randint, random, sample, shuffle
from time import sleep, time
import csv
import codecs
import json
from classes import Intersection, Endpoint, Outage, Car, Route, Routes
from vehicle_rtree import VehicleNode
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
    outage_objects.append(Outage(i, float(row['Latitude']), float(row['Longitude']), intersection_dictionary[7440], row['TimeStampCrawled'], prediction_times[i][0], prediction_times[i][1], int(row['Zip Code Ranking']), float(row['TimeStampCrawled'])))
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

def get_distance_to_outages(routes, data, outage_index, outages, zip_code_ranking_dictionary):
  def get_time(elem):
    return elem[0].max_time

  new_routes = []
  for route in routes:
    for i in range(len(route.paths)):
      new_paths = []
      total_time = 0.0
      path_time = 0.0
      for j in range(len(route.paths)):
        if i == j:
          new_paths.append(Route(route.paths[i].vehicle_id, outage_index, route.paths[i].previous_indices + [outage_index], route.paths[i].route_time + data[route.paths[i].current_index][outage_index]))
          total_time += data[route.paths[i].current_index][outage_index]
          path_time = route.paths[i].route_time + data[route.paths[i].current_index][outage_index]
        else:
          new_paths.append(route.paths[j])
        total_time += route.paths[j].route_time
      new_routes.append([Routes(new_paths, total_time), path_time])
  new_routes.sort(key=get_time)
  path_time = new_routes[0][1]
  new_routes = [new_routes[0][0]]
  
  if outages[outage_index - 1].zip_code_ranking in zip_code_ranking_dictionary:
    zip_code_ranking_dictionary[outages[outage_index - 1].zip_code_ranking][0] += path_time
    zip_code_ranking_dictionary[outages[outage_index - 1].zip_code_ranking][1] += 1
  else:
    zip_code_ranking_dictionary[outages[outage_index - 1].zip_code_ranking] = [path_time, 1]
  return new_routes
  

def insertion_based_routing(data, outages, zip_code_ranking_dictionary):
  start = time()

  starting_vehicles = [Route(index, len(outages) + 1 + index, [len(outages) + 1 + index], 0.0) for index in range(len(data) - len(outages) - 1)]
  routes = [Routes(starting_vehicles, 0.0)]
  for outage_index in range(1, len(outages) + 1):
    routes = get_distance_to_outages(routes, data, outage_index, outages, zip_code_ranking_dictionary)
  
  end = time()
  difference = end - start
  return print_info(routes[0], difference)

def get_rtree_routes(rtree, route, outage_index, outage, rtree_zip_code_ranking_times):
  outage_penalty = 0.5

  new_paths = []
  total_time = 0.0
  path_time = 0.0
  closest_vehicle = rtree.travel_to_minimum_outage(rtree, route, outage, True)
  for i in range(len(route.paths)):
    if route.paths[i].vehicle_id == closest_vehicle[1]:
      new_paths.append(Route(route.paths[i].vehicle_id, outage_index, route.paths[i].previous_indices + [outage_index], closest_vehicle[0] + outage_penalty))
      total_time += closest_vehicle[0] - route.paths[i].route_time + outage_penalty
      path_time = closest_vehicle[0] + outage_penalty
    else:
      new_paths.append(route.paths[i])
    total_time += route.paths[i].route_time

  if outage.zip_code_ranking in rtree_zip_code_ranking_times:
    rtree_zip_code_ranking_times[outage.zip_code_ranking][0] += path_time
    rtree_zip_code_ranking_times[outage.zip_code_ranking][1] += 1
  else:
    rtree_zip_code_ranking_times[outage.zip_code_ranking] = [path_time, 1]
  return Routes(new_paths, total_time)

def rtree_based_routing(data, outages, vehicles, rtree_zip_code_ranking_times):
  start = time()

  starting_vehicles = [Route(index, len(outages) + 1 + index, [len(outages) + 1 + index], 0.0) for index in range(len(data) - len(outages) - 1)]
  route = Routes(starting_vehicles, 0.0)
  rtree = VehicleNode(True, 0)
  for vehicle in vehicles:
    rtree.insert_new_value(vehicle)

  for outage_index in range(len(outages)):
    route = get_rtree_routes(rtree, route, outage_index, outages[outage_index], rtree_zip_code_ranking_times)
  
  end = time()
  difference = end - start
  return print_info(route, difference)

def print_info(route, time_difference):
  times = [path.route_time for path in route.paths]
  outages = [len(path.previous_indices) - 1 for path in route.paths]

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
  rtree = VehicleNode(True, 0)
  for outage in outages:
    rtree.insert_new_value(outage)
  return rtree

def main():
  def get_time_stamp_crawled(elem):
    return elem.time_stamp_crawled

  total_return_object = []
  outages = get_outages()
  intersection_dictionary = get_routing_data()
  prediction_times = predict_fix_times("actual", outages)
  outages = define_outages(intersection_dictionary, outages, prediction_times)
  outages_in_order = list.copy(outages)
  outages_in_order.sort(key=get_time_stamp_crawled)
  for number_of_cars in range(20, 220, 20): # change this
    rtree_zip_code_ranking_times = { }
    baseline_zip_code_ranking_times = { }
    weighted_baseline_zip_code_ranking_times = { }
    return_object = { }
    for i in range(3):
      starting_intersections = sample(list(intersection_dictionary.values()), number_of_cars)
      vehicles = [Car(index, starting_intersections[index].latitude, starting_intersections[index].longitude, starting_intersections[index]) for index in range(len(starting_intersections))]
      i_vehicles = list.copy(vehicles)
      w_vehicles = list.copy(vehicles)
      r_vehicles = list.copy(vehicles)
      w_outages = list.copy(outages)
      r_outages = list.copy(outages)

      #print(str(number_of_cars) + " before insertion")
      distance_matrix = get_distance_matrix([None] + outages_in_order + i_vehicles, intersection_dictionary)
      insertion_return_object = insertion_based_routing(distance_matrix, outages_in_order, baseline_zip_code_ranking_times)

      #print("\nbefore weighted insertion")
      distance_matrix = get_distance_matrix([None] + w_outages + w_vehicles, intersection_dictionary)
      weighted_insertion_return_object = insertion_based_routing(distance_matrix, w_outages, weighted_baseline_zip_code_ranking_times)
      
      #print("\nbefore rtree based insertion")
      rtree_return_object = rtree_based_routing(distance_matrix, r_outages, r_vehicles, rtree_zip_code_ranking_times)
      return_object[i] = insertion_return_object + rtree_return_object + weighted_insertion_return_object + [number_of_cars]

    json_suffix = '1.json'
    f = open('./baseline' + str(number_of_cars) + json_suffix, 'w')
    json.dump(baseline_zip_code_ranking_times, f)
    f = open('./weighted_baseline' + str(number_of_cars) + json_suffix, 'w')
    json.dump(weighted_baseline_zip_code_ranking_times, f)
    f = open('./rtree' + str(number_of_cars) + json_suffix, 'w')
    json.dump(rtree_zip_code_ranking_times, f)
    total_return_object.append(return_object)
  
  print(json.dumps(total_return_object))

main()