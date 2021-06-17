import distances
from classes import Node

def get_shortest_path_greedy_best_first(intersection_dictionary, starting_key, ending_key):
  initial_distance = distances.euclidean_distance(intersection_dictionary[starting_key].latitude, intersection_dictionary[starting_key].longitude, intersection_dictionary[ending_key].latitude, intersection_dictionary[ending_key].longitude)
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
        distance = distances.euclidean_distance(intersection_dictionary[endpoint].latitude, intersection_dictionary[endpoint].longitude, intersection_dictionary[ending_key].latitude, intersection_dictionary[ending_key].longitude)
        if len(priority_queue) == 0:
          priority_queue.insert(0, Node(current_node, intersection_dictionary[endpoint], distance))
        else:
          index = 0
          for i in range(len(priority_queue)):
            if distance > priority_queue[i].heuristic_time_to_goal:
              index = i + 1
          priority_queue.insert(index, Node(current_node, intersection_dictionary[endpoint], distance))
  return 'no shortest path'