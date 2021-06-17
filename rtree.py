class Value:
  def __init__(self, latitude, longitude, is_car, id=-1):
    self.latitude = latitude
    self.longitude = longitude
    self.is_car = is_car
    self.id = id
  def convert_to_dict(self):
    return { "latitude": self.latitude, "longitude": self.longitude, "is_car": self.is_car, "vehicle_id": self.vehicle_id }

class Node:
  def __init__(self, is_leaf, index):
    self.nodes = []
    self.is_leaf = is_leaf
    self.index = index
    self.values = []
    self.min_number_of_values = 2
    self.max_number_of_values = 6
    self.number_of_vehicles = 0
    self.minimum_latitude = None
    self.minimum_longitude = None
    self.maximum_latitude = None
    self.maximum_longitude = None
    self.area = 0.0

  def convert_to_dict(self):
    return { "minimum_latitude": self.minimum_latitude, "maximum_latitude": self.maximum_latitude, "minimum_longitude": self.minimum_longitude,
      "maximum_longitude": self.maximum_longitude, "values": self.values }
  
  def search(self, node, outage):
    if node.is_leaf:
      for value in node.values:
        if value.latitude == outage.latitude and value.longitude == outage.longitude:
          return nodes.index
      return -1
    else:
      index = -1
      for temp_node in node.nodes:
        if self.within_boundaries(outage):
          index = self.search(temp_node, outage)
      return index

  def new_theoretical_area(self, outage):
    if self.within_boundaries(outage):
      return self.area
    temp_minimum_latitude = self.minimum_latitude
    temp_minimum_longitude = self.minimum_longitude
    temp_maximum_latitude = self.maximum_latitude
    temp_maximum_longitude = self.maximum_longitude
    
    if self.minimum_latitude == None or outage.latitude < self.minimum_latitude:
      temp_minimum_latitude = outage.latitude
    if self.maximum_latitude == None or outage.latitude > self.maximum_latitude:
      temp_maximum_latitude = outage.latitude
    if self.minimum_longitude == None or outage.longitude < self.minimum_longitude:
      temp_minimum_longitude = outage.longitude
    if self.maximum_longitude == None or outage.longitude > self.maximum_longitude:
      temp_maximum_longitude = outage.longitude
    return (temp_maximum_latitude - temp_minimum_latitude) * (temp_maximum_longitude - temp_minimum_longitude)
  
  def split(self, k_constant):
    values = []
    nodes = [self]
    while len(nodes) > 0:
      if nodes[0].is_leaf:
        for value in nodes[0].values:
          values.append(value)
        self.reset_node(nodes[0])
        nodes[0].nodes.append(Node(True, nodes[0].index + 1))
        nodes[0].nodes.append(Node(True, nodes[0].index + 2))
        nodes.pop(0)
      else:
        self.reset_node(nodes[0])
        nodes.append(nodes[0].nodes[0])
        nodes.append(nodes[0].nodes[1])
        nodes.pop(0)
    for value in values:
      self.insert_new_value(value, k_constant)

  def set_new_nodes_boundaries(self, outage):
    if self.minimum_latitude == None or outage.latitude < self.minimum_latitude:
      self.minimum_latitude = outage.latitude
    if self.maximum_latitude == None or outage.latitude > self.maximum_latitude:
      self.maximum_latitude = outage.latitude
    if self.minimum_longitude == None or outage.longitude < self.minimum_longitude:
      self.minimum_longitude = outage.longitude
    if self.maximum_longitude == None or outage.longitude > self.maximum_longitude:
      self.maximum_longitude = outage.longitude
    self.area = (self.maximum_latitude - self.minimum_latitude) * (self.maximum_longitude - self.minimum_longitude)

  def insert_deeper(self, node, outage, max_number_achieved, k_constant):
    if node.is_leaf:
      if len(node.values) == node.max_number_of_values:
        return -1
      elif max_number_achieved and len(node.values) >= node.min_number_of_values:
        return -1
      else:
        if outage.is_car:
          node.number_of_vehicles += 1
          node.values.append(Value(outage.latitude, outage.longitude, outage.is_car, outage.id))
        else:
          node.values.append(Value(outage.latitude, outage.longitude, outage.is_car))
        closest_car_node = None
        if node.number_of_vehicles >= k_constant:
          closest_car_node = node
        return closest_car_node
    else:
      def get_theoretical_area(elem):
        return elem[1]
      areas = [(temp_node, temp_node.new_theoretical_area(outage)) for temp_node in node.nodes]
      areas.sort(key=get_theoretical_area)
      max_number_achieved = False
      for area in areas:
        insertion_return = self.insert_deeper(area[0], outage, max_number_achieved, k_constant)
        if insertion_return == -1:
          max_number_achieved = True
        else:
          area[0].set_new_nodes_boundaries(outage)
          if outage.is_car:
            area[0].number_of_vehicles += 1
          if area[0].number_of_vehicles >= k_constant and insertion_return == None:
            insertion_return = area[0]
          return insertion_return
      self.split(k_constant)
      return self.insert_new_value(outage, k_constant)

  def insert_new_value(self, outage, k_constant):
    if self.search(self, outage) != -1:
      return False
    insertion_return = self.insert_deeper(self, outage, False, k_constant)
    if insertion_return == -1:
      self.split(k_constant)
      insertion_return = self.insert_deeper(self, outage, False, k_constant)
    if insertion_return == None:
      return self
    return insertion_return
  
  def within_boundaries(self, outage):
    if self.area == 0:
      return False
    if outage.latitude >= self.minimum_latitude and outage.latitude <= self.maximum_latitude and outage.longitude >= self.minimum_longitude and outage.longitude <= self.maximum_longitude:
      return True
    return False

  def return_all_values(self, node):
    if node.is_leaf:
      return node.values
    else:
      all_values = []
      for temp_node in node.nodes:
        all_values = all_values + temp_node.return_all_values(temp_node)
      return all_values

  def reset_node(self, node):
    node.is_leaf = False
    node.values = []
    node.number_of_vehicles = 0
    node.minimum_latitude = None
    node.minimum_longitude = None
    node.maximum_latitude = None
    node.maximum_longitude = None
    node.area = 0.0