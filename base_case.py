from distances import euclidean_distance
from classes import VehicleShortestRouteInfo

def get_closet_vehicle_index(data, outages, vehicles, index):
  min_vehicle_index = -1
  min_vehicle_duration = -1
  for vehicle in vehicles:
    if (min_vehicle_index == -1 and vehicle[1] <= 0.0) or data[vehicle[2]][index] < min_vehicle_duration:
      min_vehicle_index = vehicle[5]
      min_vehicle_duration = data[vehicle[2]][index]
  return min_vehicle_index

def vehicular_routing_problem_base_case(data, outages):
  vehicles = [[(len(outages) + 1 + index), 0.0, (len(outages) + 1 + index), [(len(outages) + 1 + index)], 0.0, index] for index in range(len(data) - (len(outages) + 1))]
  for i in range(len(outages)):
    min_vehicle_index = -1
    while min_vehicle_index == -1:
      min_vehicle_index = get_closet_vehicle_index(data, outages, vehicles, i + 1)
      if min_vehicle_index == -1:
        for vehicle in vehicles:
          vehicle[1] -= 50000
          vehicle[4] += 50000
    vehicles[min_vehicle_index][1] += data[vehicles[min_vehicle_index][2]][i + 1] + outages[i].actual_fix_time * 60000
    vehicles[min_vehicle_index][2] = i + 1
    vehicles[min_vehicle_index][3].append(i + 1)
    vehicles[min_vehicle_index][4] += outages[i].actual_fix_time * 60000

  info = []
  for vehicle in vehicles:
    info.append(VehicleShortestRouteInfo(vehicle[0], vehicle[3], vehicle[4], vehicle[4]))
  return info

    

