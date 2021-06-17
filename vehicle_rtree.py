class Vehicle:
  def __init__(self, latitude, longitude, id=-1):
    self.latitude = latitude
    self.longitude = longitude
    self.id = id
  def convert_to_dict(self):
    return { "latitude": self.latitude, "longitude": self.longitude, "vehicle_id": self.vehicle_id }

class VehicleNode:
  def __init__(self, is_leaf, index):
    self.nodes = []
    self.is_leaf = is_leaf
    self.index = index
    self.vehicles = []
    self.min_number_of_values = 6
    self.max_number_of_values = 10
    self.number_of_vehicles = 0
    self.minimum_latitude = None
    self.minimum_longitude = None
    self.maximum_latitude = None
    self.maximum_longitude = None
    self.area = 0.0

  def convert_to_dict(self):
    return { "minimum_latitude": self.minimum_latitude, "maximum_latitude": self.maximum_latitude, "minimum_longitude": self.minimum_longitude,
      "maximum_longitude": self.maximum_longitude, "vehicles": self.vehicles }

  def euclidean_distance(self, outage, vehicle):
    return (((outage.latitude - vehicle.latitude) ** 2) + ((outage.longitude - vehicle.longitude) ** 2)) ** (0.5)

  def search(self, node, vehicle):
    if node.is_leaf:
      for v in node.vehicles:
        if v.id == vehicle.id:
          return v
      return None
    else:
      index = None
      for temp_node in node.nodes:
        if self.within_boundaries(vehicle):
          index = self.search(temp_node, vehicle)
          if index != None:
            return index
      return None

  def new_theoretical_area(self, vehicle):
    if self.within_boundaries(vehicle):
      return self.area
    temp_minimum_latitude = self.minimum_latitude
    temp_minimum_longitude = self.minimum_longitude
    temp_maximum_latitude = self.maximum_latitude
    temp_maximum_longitude = self.maximum_longitude
    
    if self.minimum_latitude == None or vehicle.latitude < self.minimum_latitude:
      temp_minimum_latitude = vehicle.latitude
    if self.maximum_latitude == None or vehicle.latitude > self.maximum_latitude:
      temp_maximum_latitude = vehicle.latitude
    if self.minimum_longitude == None or vehicle.longitude < self.minimum_longitude:
      temp_minimum_longitude = vehicle.longitude
    if self.maximum_longitude == None or vehicle.longitude > self.maximum_longitude:
      temp_maximum_longitude = vehicle.longitude
    return (temp_maximum_latitude - temp_minimum_latitude) * (temp_maximum_longitude - temp_minimum_longitude)

  def split(self):
    vehicles = []
    nodes = [self]
    while len(nodes) > 0:
      if nodes[0].is_leaf:
        for vehicle in nodes[0].vehicles:
          vehicles.append(vehicle)
        self.reset_node(nodes[0])
        nodes[0].nodes.append(VehicleNode(True, nodes[0].index + 1))
        nodes[0].nodes.append(VehicleNode(True, nodes[0].index + 2))
        nodes.pop(0)
      else:
        self.reset_node(nodes[0])
        nodes.append(nodes[0].nodes[0])
        nodes.append(nodes[0].nodes[1])
        nodes.pop(0)
    for vehicle in vehicles:
      self.insert_new_value(vehicle)

  def set_new_nodes_boundaries(self, vehicle):
    if self.minimum_latitude == None or vehicle.latitude < self.minimum_latitude:
      self.minimum_latitude = vehicle.latitude
    if self.maximum_latitude == None or vehicle.latitude > self.maximum_latitude:
      self.maximum_latitude = vehicle.latitude
    if self.minimum_longitude == None or vehicle.longitude < self.minimum_longitude:
      self.minimum_longitude = vehicle.longitude
    if self.maximum_longitude == None or vehicle.longitude > self.maximum_longitude:
      self.maximum_longitude = vehicle.longitude
    self.area = (self.maximum_latitude - self.minimum_latitude) * (self.maximum_longitude - self.minimum_longitude)

  def insert_deeper(self, node, vehicle, max_number_achieved):
    if node.is_leaf:
      if len(node.vehicles) == node.max_number_of_values:
        return -1
      elif max_number_achieved and len(node.vehicles) >= node.min_number_of_values:
        return -1
      else:
        node.set_new_nodes_boundaries(vehicle)
        node.number_of_vehicles += 1
        node.vehicles.append(Vehicle(vehicle.latitude, vehicle.longitude, vehicle.id))
    else:
      def get_theoretical_area(elem):
        return elem[1]
      areas = [(temp_node, temp_node.new_theoretical_area(vehicle)) for temp_node in node.nodes]
      areas.sort(key=get_theoretical_area)
      max_number_achieved = False
      for area in areas:
        insertion_return = self.insert_deeper(area[0], vehicle, max_number_achieved)
        if insertion_return == -1:
          max_number_achieved = True
        else:
          node.set_new_nodes_boundaries(vehicle)
          node.number_of_vehicles += 1
          return 1
      return -1

  def insert_new_value(self, vehicle):
    if self.search(self, vehicle) != None:
      return False
    insertion_return = self.insert_deeper(self, vehicle, False)
    if insertion_return == -1:
      self.split()
      self.insert_deeper(self, vehicle, False)
    return True
  
  def within_boundaries(self, value):
    if self.area == 0:
      return False
    if value.latitude >= self.minimum_latitude and value.latitude <= self.maximum_latitude and value.longitude >= self.minimum_longitude and value.longitude <= self.maximum_longitude:
      return True
    return False

  def return_all_values(self, node):
    if node.is_leaf:
      return node.vehicles
    else:
      all_values = []
      for temp_node in node.nodes:
        all_values = all_values + temp_node.return_all_values(temp_node)
      return all_values

  def get_rtree_vehicle_lengths(self, node):
    if node.is_leaf:
      print(len(node.vehicles))
    else:
      for temp_node in node.nodes:
        temp_node.get_rtree_vehicle_lengths(temp_node)

  def print_all_areas(self, node):
    if node.is_leaf:
      print(node.area)
    else:
      print(node.area)
      for temp_node in node.nodes:
        self.print_all_areas(temp_node)

  def reset_node(self, node):
    node.is_leaf = False
    node.vehicles = []
    node.number_of_vehicles = 0
    node.minimum_latitude = None
    node.minimum_longitude = None
    node.maximum_latitude = None
    node.maximum_longitude = None
    node.area = 0.0

  def closest_vehicle(self, node, route, vehicles, outage, should_insert):
    def get_minimum_trip(elem):
      return elem[0]

    distances_and_indices = []

    for i in range(len(vehicles)):
      for j in range(len(route.paths)):
        if route.paths[j].vehicle_id == vehicles[i].id:
          distances_and_indices.append((self.euclidean_distance(vehicles[i], outage) + route.paths[j].route_time, vehicles[i].id, i, j))
    
    closest_vehicle = min(distances_and_indices, key=get_minimum_trip)
    if should_insert:
      vehicles[closest_vehicle[2]].latitude = outage.latitude
      vehicles[closest_vehicle[2]].longitude = outage.longitude
      node.set_new_nodes_boundaries(vehicles[closest_vehicle[2]])

    return closest_vehicle

  def travel_to_minimum_outage(self, node, route, outage, should_insert):
    if node.is_leaf:
      return self.closest_vehicle(node, route, node.vehicles, outage, should_insert)
    else:
      def get_theoretical_area(elem):
        return elem[1]
      areas = [(temp_node, temp_node.new_theoretical_area(outage)) for temp_node in node.nodes]
      areas.sort(key=get_theoretical_area)
      while areas[0][0].number_of_vehicles == 0:
        areas.pop(0)
      if should_insert:
        node.set_new_nodes_boundaries(outage)
      return self.travel_to_minimum_outage(areas[0][0], route, outage, should_insert)