from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from classes import VehicleShortestRouteInfo

def create_solution_static_case(data, manager, routing, solution):
  info = []
  max_route_duration = 0
  for vehicle_id in range(data['num_vehicles']):
    index = routing.Start(vehicle_id)
    route_indices = []
    route_distance = 0
    while not routing.IsEnd(index):
      route_indices.append(manager.IndexToNode(index))
      previous_index = index
      index = solution.Value(routing.NextVar(index))
      route_distance += routing.GetArcCostForVehicle(
        previous_index, index, vehicle_id)
    route_indices.append(manager.IndexToNode(index))
    current_route_min = route_distance
    current_route_max = route_distance
    max_route_duration = max(route_distance, max_route_duration)
    info.append(VehicleShortestRouteInfo(vehicle_id, route_indices, current_route_min, current_route_max))
  return info

def vehicular_routing_problem_static_case(data):
  manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['starts'],
                                           data['ends'])
  routing = pywrapcp.RoutingModel(manager)
  def distance_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return data['distance_matrix'][from_node][to_node]

  transit_callback_index = routing.RegisterTransitCallback(distance_callback)

  routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

  dimension_name = 'Distance'
  routing.AddDimension(
    transit_callback_index,
    0,  # no slack
    9000000000,  # vehicle maximum travel distance
    True,  # start cumul to zero
    dimension_name)
  distance_dimension = routing.GetDimensionOrDie(dimension_name)
  distance_dimension.SetGlobalSpanCostCoefficient(100)

  search_parameters = pywrapcp.DefaultRoutingSearchParameters()
  search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

  solution = routing.SolveWithParameters(search_parameters)

  if solution:
    return (data, manager, routing, solution)

  return None