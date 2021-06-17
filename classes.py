import json

class Outage:
  def __init__(self, id, latitude, longitude, closest_intersection, outage_date_time, predicted_fix_time, actual_fix_time=None, zip_code_ranking=-1, time_stamp_crawled=-1):
    self.id = id
    self.latitude = latitude
    self.longitude = longitude
    self.closest_intersection = closest_intersection
    self.is_car = False
    self.outage_date_time = outage_date_time
    self.predicted_fix_time = predicted_fix_time
    self.actual_fix_time = actual_fix_time
    self.zip_code_ranking = zip_code_ranking
    self.time_stamp_crawled = time_stamp_crawled
  def convert_to_dict(self):
    return { "id": self.id, "latitude": self.latitude, "longitude": self.longitude, "closest_intersection": self.closest_intersection.convert_to_dict(), "is_car": self.is_car,
      "outage_date_time": self.outage_date_time, "predicted_fix_time": self.predicted_fix_time,  "actual_fix_time": self.actual_fix_time, "zip_code_ranking": self.zip_code_ranking }

class Route:
  def __init__(self, vehicle_id, current_index, previous_indices, route_time):
    self.vehicle_id = vehicle_id
    self.current_index = current_index
    self.previous_indices = previous_indices
    self.route_time = route_time
  def convert_to_dict(self):
    return { "vehicle_id": self.vehicle_id, "current_index": self.current_index, "route_time": self.route_time }

class Routes:
  def __init__(self, paths, max_time):
    self.paths = paths
    self.max_time = max_time

class Car:
  def __init__(self, id, latitude, longitude, closest_intersection):
    self.id = id
    self.latitude = latitude
    self.longitude = longitude
    self.closest_intersection = closest_intersection
    self.is_car = True
  def convert_to_dict(self):
    return { "id": self.id, "latitude": self.latitude, "longitude": self.longitude, "closest_intersection": self.closest_intersection.convert_to_dict(), "is_car": self.is_car }

class Intersection:
  def __init__(self, id, key, latitude, longitude):
    self.id = id
    self.key = key
    self.latitude = latitude
    self.longitude = longitude
    self.has_been_read = False
    self.endpoints = { }
  def convert_to_dict(self):
    return { "id": self.id, "key": self.key, "latitude": self.latitude, "longitude": self.longitude, "has_been_read": self.has_been_read }

class Endpoint:
  def __init__(self, key, time_to_goal):
    self.key = key
    self.time_to_goal = time_to_goal
  def convert_to_dict(self):
    return { "key": self.key, "time_to_goal": self.time_to_goal }

class Node:
  def __init__(self, previous_node, intersection, heuristic_time_to_goal):
    self.previous_node = previous_node
    self.intersection = intersection
    self.heuristic_time_to_goal = heuristic_time_to_goal
  def convert_to_dict(self):
    return { "intersection": self.intersection.convert_to_dict(), "heuristic_time_to_goal": self.heuristic_time_to_goal }

class Centroid:
  def __init__(self, key, latitude, longitude):
    self.key = key
    self.latitude = latitude
    self.longitude = longitude
  def convert_to_dict(self):
    return { "key": self.key, "latitude": self.latitude, "longitude": self.longitude }

class Cluster:
  def __init__(self, centroid, outage):
    self.centroid = centroid
    self.outages = [outage]
    self.vehicles = []
  def convert_to_dict(self):
    return { "centroid": self.centroid.convert_to_dict(), "outages": [out.convert_to_dict() for out in self.outages], "vehicles": [veh.convert_to_dict() for veh in self.vehicles] }

class CentroidDifference:
  def __init__(self):
    self.latitude_difference = 0.0
    self.longitude_difference = 0.0
    self.number_of_differences = 0
  def convert_to_dict(self):
    return { "latitude_difference": self.latitude_difference, "longitude_difference": self.longitude_difference, "number_of_differences": self.number_of_differences }

class VehicleShortestRouteInfo:
  def __init__(self, vehicle_id, route_indices, minimum_time, maximum_time):
    self.vehicle_id = vehicle_id
    self.route_indices = route_indices
    self.minimum_time = minimum_time
    self.maximum_time = maximum_time
  def convert_to_dict(self):
    return { "vehicle_id": self.vehicle_id, "route_indices": self.route_indices, "minimum_time": self.minimum_time, "maximum_time": self.maximum_time }