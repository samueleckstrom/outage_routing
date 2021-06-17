from math import sqrt, pow
import pathfinding

def euclidean_distance(x1, y1, x2, y2):
  return sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2))

def closest_centroid(latitude, longitude, centroids):
  min_distance = float('inf')
  min_latitude_difference = 0.0
  min_longitude_difference = 0.0
  key = -1
  for centroid in centroids:
    (latitude_difference, longitude_difference, current_distance) = determine_centroid_distances(latitude, longitude, centroid)
    if current_distance < min_distance:
      key = centroid.key
      min_distance = current_distance
      min_latitude_difference = latitude_difference
      min_longitude_difference = longitude_difference
  return (key, min_latitude_difference, min_longitude_difference, min_distance)

def get_distance_matrix(outages_and_cars, intersection_dictionary, type_of_matrix):
  distance_matrix = [[0.0 for index in range(len(outages_and_cars))] for index in range(len(outages_and_cars))] # need to do something here
  routing_matrix = [[None for index in range(len(outages_and_cars))] for index in range(len(outages_and_cars))]
  for i in range(len(outages_and_cars)):
    for j in range(len(outages_and_cars)):
      path_distance = None
      path_intersections = []
      if i == j or i == 0 or j == 0:
        path_distance = 0.0
      elif outages_and_cars[j].is_car: # this was changed
        path_distance = 10000000000.0
      else:
        for intersection in intersection_dictionary.values():
          intersection.has_been_read = False
        path = pathfinding.get_shortest_path_greedy_best_first(intersection_dictionary, outages_and_cars[i].closest_intersection.key,
          outages_and_cars[j].closest_intersection.key)
        if path == 'no shortest path':
          path_distance = 10000000000.0
        else:
          if type_of_matrix == 'prediction':
            path_distance = outages_and_cars[j].predicted_fix_time * 60000.0
          elif type_of_matrix == 'actual':
            path_distance = outages_and_cars[j].actual_fix_time * 60000.0
          else:
            path_distance = 0.0
          current_path = path
          while current_path.previous_node != None:
            path_intersections.append([current_path.intersection.latitude, current_path.intersection.longitude])
            path_distance += current_path.intersection.endpoints[current_path.previous_node.intersection.key].time_to_goal
            current_path = current_path.previous_node
      
      distance_matrix[i][j] = path_distance
      routing_matrix[i][j] = path_intersections
  return (distance_matrix, routing_matrix)

def determine_centroid_distances(latitude, longitude, centroid):
  latitude_difference = latitude - centroid.latitude
  longitude_difference = longitude - centroid.longitude
  distance = euclidean_distance(latitude, longitude, centroid.latitude, centroid.longitude)
  return (latitude_difference, longitude_difference, distance)
