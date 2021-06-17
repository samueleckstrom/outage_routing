from math import sqrt

def get_distance_matrix(outages_and_cars, intersection_dictionary):
  distance_matrix = [[0.0 for index in range(len(outages_and_cars))] for index in range(len(outages_and_cars))] # need to do something here
  for i in range(len(outages_and_cars)):
    for j in range(len(outages_and_cars)):
      path_distance = None
      if i == j or i == 0 or j == 0:
        path_distance = 0.0
      elif outages_and_cars[j].is_car: # this was changed
        path_distance = 10000000000.0
      else:
        path_distance = sqrt(pow(outages_and_cars[i].latitude - outages_and_cars[j].latitude, 2)
          + pow(outages_and_cars[i].longitude - outages_and_cars[j].longitude, 2)) + 0.5
      
      distance_matrix[i][j] = path_distance
  return distance_matrix