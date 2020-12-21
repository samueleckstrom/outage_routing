from ortools.sat.python import cp_model
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import math
from random import randint, random, sample
from time import sleep
import csv
import codecs

class Outage:
  def __init__(self, id, latitude, longitude, closest_intersection):
    self.id = id
    self.latitude = latitude
    self.longitude = longitude
    self.closest_intersection = closest_intersection

class Intersection:
  def __init__(self, id, key, latitude, longitude):
    self.id = id
    self.key = key
    self.latitude = latitude
    self.longitude = longitude
    self.has_been_read = False
    self.endpoints = { }

class Endpoint:
  def __init__(self, key, time_to_goal):
    self.key = key
    self.time_to_goal = time_to_goal

class Node:
  def __init__(self, previous_node, intersection, heuristic_time_to_goal):
    self.previous_node = previous_node
    self.intersection = intersection
    self.heuristic_time_to_goal = heuristic_time_to_goal

class Centroid:
  def __init__(self, key, latitude, longitude):
    self.key = key
    self.latitude = latitude
    self.longitude = longitude

class Cluster:
  def __init__(self, centroid, outage):
    self.centroid = centroid
    self.outages = [outage]
    self.vehicles = []

def get_routing_data():
  intersection_dictionary = { }

  f = open('./USA-road-d.E.co', 'r')
  id = 0
  for row in f:
    words = row.split(' ')
    if len(words) == 4 and int(words[2]) < -73700000 and int(words[2]) > -74300000 and int(words[3]) > 40500000 and int(words[3]) < 41000000:
      longitude = words[2] = float(words[2]) / 1000000.0
      latitude = float(words[3]) / 1000000.0
      intersection_dictionary[int(words[1])] = Intersection(id, int(words[1]), latitude, longitude)
      id += 1
  
  f = open('./USA-road-t.E.gr', 'r')
  for row in f:
    words = row.split(' ')
    if len(words) == 4 and words[0] == 'a' and int(words[1]) in intersection_dictionary and int(words[2]) in intersection_dictionary:
      intersection_dictionary[int(words[1])].endpoints[int(words[2])] = Endpoint(int(words[2]), int(words[3]))
  return intersection_dictionary

def determine_intersection_distance(outage_latitude, outage_longitude, intersection):
    return math.sqrt(math.pow((intersection.latitude - outage_latitude), 2) + math.pow((intersection.longitude - outage_longitude), 2))

def find_closest_intersections(outages, intersection_dictionary):
  for outage in outages:
    for intersection in intersection_dictionary.values():
      current_outage_distance = determine_intersection_distance(outage.latitude, outage.longitude, outage.closest_intersection)
      potential_outage_distance = determine_intersection_distance(outage.latitude, outage.longitude, intersection)
      if potential_outage_distance < current_outage_distance:
        outage.closest_intersection = intersection

def determine_distance(start_intersection, end_intersection):
  return math.sqrt(math.pow((start_intersection.latitude - end_intersection.latitude), 2) + math.pow((start_intersection.longitude - end_intersection.longitude), 2))

def get_shortest_path_greedy_best_first(intersection_dictionary, starting_key, ending_key):
  initial_distance = determine_distance(intersection_dictionary[starting_key], intersection_dictionary[ending_key])
  starting_point = Node(None, intersection_dictionary[starting_key], initial_distance)
  priority_queue = [starting_point]
  while len(priority_queue) > 0:
    current_node = priority_queue.pop(0)
    if current_node.intersection.has_been_read:
      continue
    else:
      current_node.intersection.has_been_read = True
    if current_node.intersection.key == ending_key:
      return current_node
    else:
      for endpoint in current_node.intersection.endpoints.keys():
        if intersection_dictionary[endpoint].has_been_read:
          continue  # dont go backwards... infinite loopy
        distance = determine_distance(intersection_dictionary[endpoint], intersection_dictionary[ending_key])
        if len(priority_queue) == 0:
          priority_queue.insert(0, Node(current_node, intersection_dictionary[endpoint], distance))
        else:
          index = 0
          for i in range(len(priority_queue)):
            if distance > priority_queue[i].heuristic_time_to_goal:
              index = i + 1
          priority_queue.insert(index, Node(current_node, intersection_dictionary[endpoint], distance))
  return 'no shortest path'

def get_distance_matrix(outages, intersection_dictionary):
  distance_matrix = [[0.0 for i in range(len(outages))] for j in range(len(outages))]
  for i in range(len(outages)):
    for j in range(len(outages)):
      if i == j:
        path_distance = 0.0
      else:
        for intersection in intersection_dictionary.values():
          intersection.has_been_read = False
        print(i * j)
        path = get_shortest_path_greedy_best_first(intersection_dictionary, outages[i].closest_intersection.key,
          outages[j].closest_intersection.key)
        if path == 'no shortest path':
          path_distance = 1000000000.0
        else:
          path_distance = 0
          current_path = path
          while current_path.previous_node != None:
            path_distance += current_path.intersection.endpoints[current_path.previous_node.intersection.key].time_to_goal
            current_path = current_path.previous_node    
      distance_matrix[i][j] = path_distance
  return distance_matrix

def determine_centroid_distances(latitude, longitude, centroid):
  latitude_difference = latitude - centroid.latitude
  longitude_difference = longitude - centroid.longitude
  distance = math.sqrt(math.pow((latitude - centroid.latitude), 2) + math.pow((longitude - centroid.longitude), 2))
  return (latitude_difference, longitude_difference, distance)

def find_closest_centroid(latitude, longitude, list_of_centroids):
  min_distance = float('inf')
  min_latitude_difference = 0.0
  min_longitude_difference = 0.0
  key = -1
  for centroid in list_of_centroids:
    (latitude_difference, longitude_difference, current_distance) = determine_centroid_distances(latitude, longitude, centroid)
    if current_distance < min_distance:
      key = centroid.key
      min_distance = current_distance
      min_latitude_difference = latitude_difference
      min_longitude_difference = longitude_difference
  return (key, min_distance, min_latitude_difference, min_longitude_difference)

def k_means(intersection_dictionary, number_of_outages, number_of_k_means_iterations, number_of_centroids):
  rows = []
  test_matrix = []
  with open('./historical_outages.csv', 'r') as csvfile:
    data = csv.reader(csvfile, delimiter=',')
    for row in data:
      rows.append(row)
  rows.pop(0)

  for i in range(number_of_k_means_iterations):
    rows_copy = rows.copy()
    test_row = []
    for j in range(number_of_outages):
      row = rows_copy.pop(randint(0, len(rows_copy) - 1))
      test_row.append(Outage(j, float(row[3]), float(row[4]), intersection_dictionary[2042109]))
    test_matrix.append(test_row)
    find_closest_intersections(test_matrix[i], intersection_dictionary)
  
  random_intersections = get_random_intersections(intersection_dictionary, number_of_centroids)
  k_means_centroids = [Centroid(intersection_index, random_intersections[intersection_index][1].latitude, random_intersections[intersection_index][1].longitude) for intersection_index in range(len(random_intersections))]
  for centroid in k_means_centroids:
    print(centroid)

  clusters = []
  for i in range(number_of_k_means_iterations):
    for j in range(100):
      k_means_differences = []
      for k in range(number_of_outages):
        k_means_differences.append(find_closest_centroid(test_matrix[i][k].latitude, test_matrix[i][k].longitude, k_means_centroids))
      latitude_difference = [0.0 for i in range(number_of_centroids)]
      longitude_difference = [0.0 for i in range(number_of_centroids)]
      number_of_differences = [0 for i in range(number_of_centroids)]
      for differences in k_means_differences:
        latitude_difference[differences[0]] += differences[2]
        longitude_difference[differences[0]] += differences[3]
        number_of_differences[differences[0]] += 1
      for k in range(number_of_centroids):
        latitude_difference[k] = latitude_difference[k] / number_of_differences[k] if number_of_differences[k] != 0 else 0.0
        longitude_difference[k] = longitude_difference[k] / number_of_differences[k] if number_of_differences[k] != 0 else 0.0
        k_means_centroids[k].latitude += latitude_difference[k]
        k_means_centroids[k].longitude += longitude_difference[k]
    
    single_layer_clusters = { }
    for j in range(number_of_outages):
      (key, _, _, _) = find_closest_centroid(test_matrix[i][j].latitude, test_matrix[i][j].longitude, k_means_centroids)
      if key in single_layer_clusters:
        single_layer_clusters[key].outages.append(test_matrix[i][j])
      else:
        single_layer_clusters[key] = Cluster(k_means_centroids[key], test_matrix[i][j])
    clusters.append(single_layer_clusters)
  return (clusters, k_means_centroids)
  
def get_random_intersections(intersection_dictionary, number_of_intersections = 10):
  return sample(intersection_dictionary.items(), number_of_intersections)

def create_vehicle_routing_data(list_of_clusters, k_means_centroids, list_of_starting_intersections):
  for starting_intersection in list_of_starting_intersections:
    (key, _, _, _) = find_closest_centroid(starting_intersection[1].latitude, starting_intersection[1].longitude, k_means_centroids)
    list_of_clusters[key].vehicles.append(starting_intersection)
    # list_of_clusters[key].outages.append(Outage(-1, starting_intersection[1].latitude, starting_intersection[1].longitude, starting_intersection[1]))

def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    max_route_distance = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} -> '.format(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += '{}\n'.format(manager.IndexToNode(index))
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        print(plan_output)
        max_route_distance = max(route_distance, max_route_distance)
    print('Maximum of the route distances: {}m'.format(max_route_distance))
    # [END solution_printer]


def vehicular_routing_problem(data):
  print('here')
  # Create the routing index manager.
  # [START index_manager]
  manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['starts'])
  # [END index_manager]

  # Create Routing Model.
  # [START routing_model]
  routing = pywrapcp.RoutingModel(manager)

  # Create and register a transit callback.
  # [START transit_callback]
  def distance_callback(from_index, to_index):
    """Returns the distance between the two nodes."""
    # Convert from routing variable Index to distance matrix NodeIndex.
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return data['distance_matrix'][from_node][to_node]

  transit_callback_index = routing.RegisterTransitCallback(distance_callback)
  # [END transit_callback]

  # Define cost of each arc.
  # [START arc_cost]
  routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
  # [END arc_cost]

  # Add Distance constraint.
  # [START distance_constraint]
  dimension_name = 'Distance'
  routing.AddDimension(
      transit_callback_index,
      0,  # no slack
      2000,  # vehicle maximum travel distance
      True,  # start cumul to zero
      dimension_name)
  distance_dimension = routing.GetDimensionOrDie(dimension_name)
  distance_dimension.SetGlobalSpanCostCoefficient(100)
  # [END distance_constraint]

  # Setting first solution heuristic.
  # [START parameters]
  search_parameters = pywrapcp.DefaultRoutingSearchParameters()
  search_parameters.first_solution_strategy = (
      routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
  # [END parameters]

  # Solve the problem.
  # [START solve]
  solution = routing.SolveWithParameters(search_parameters)
  # [END solve]

  # Print solution on console.
  # [START print_solution]
  if solution:
    print_solution(data, manager, routing, solution)
  # [END print_solution]

def main():
  intersection_dictionary = get_routing_data()
  number_of_k_means_iterations = 1
  number_of_outages = 100
  for iteration in range(number_of_k_means_iterations):
    for number_of_centroids in range(number_of_outages - 2):
      (list_of_clusters, k_means_centroids) = k_means(intersection_dictionary, number_of_outages, number_of_k_means_iterations, number_of_centroids + 2)
      list_of_starting_intersections = get_random_intersections(intersection_dictionary, number_of_intersections = 100)
      create_vehicle_routing_data(list_of_clusters[0], k_means_centroids, list_of_starting_intersections)
      distance_matrix = []
      for clusters in list_of_clusters:
        for cluster in clusters.values():
          data = { }
          print('before distance')
          data['distance_matrix'] = get_distance_matrix(cluster.outages, intersection_dictionary)
          print('after distance')
          data['num_vehicles'] = len(list_of_starting_intersections)
          data['starts'] = [randint(0, len(cluster.outages)) for i in range(len(list_of_starting_intersections))]
          vehicular_routing_problem(data)

main()