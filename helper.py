from classes import Intersection, Endpoint, Outage
from distances import euclidean_distance
import csv

def get_routing_data():
  intersection_dictionary = { }

  f = open('./USA-road-d.NY.co', 'r')
  id = 0
  for row in f:
    words = row.split(' ')
    if len(words) != 4:
      continue
    elif int(words[2]) < -74058740 and int(words[3]) > 40654109:
      continue
    elif int(words[2]) < -74014226 and int(words[3]) > 40758656:
      continue
    if int(words[2]) < -73750000 and int(words[2]) > -74260000 and int(words[3]) > 40450000 and int(words[3]) < 40950000:
      longitude = words[2] = float(words[2]) / 1000000.0
      latitude = float(words[3]) / 1000000.0
      intersection_dictionary[int(words[1])] = Intersection(id, int(words[1]), latitude, longitude)
      id += 1
  
  f = open('./USA-road-t.NY.gr', 'r')
  for row in f:
    words = row.split(' ')
    if len(words) == 4 and words[0] == 'a' and int(words[1]) in intersection_dictionary and int(words[2]) in intersection_dictionary:
      intersection_dictionary[int(words[1])].endpoints[int(words[2])] = Endpoint(int(words[2]), int(words[3]))
  
  with open('intersections.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in intersection_dictionary.values():
      spamwriter.writerow([row.id, row.key, row.latitude, row.longitude])
  return intersection_dictionary

def find_closest_intersections(intersection_dictionary):
  outages = []
  f = open('historical_outages.csv', 'r')
  rows = csv.reader(f, delimiter=',')
  i = 0
  for row in rows:
    if row[3] == 'Latitude':
      continue
    outages.append(Outage(i, float(row[3]), float(row[4]), intersection_dictionary[7440]))
    i += 1
  i = 0
  f = open('outages.csv', 'w')
  writer = csv.writer(f, delimiter=',')
  for outage in outages:
    if not (float(outage.longitude) < -73.750000 and float(outage.longitude) > -74.260000 and float(outage.latitude) > 40.450000 and float(outage.latitude) < 40.950000):
      continue
    elif float(outage.longitude) < -74.058740 and float(outage.latitude) > 40.654109:
      continue
    print(i)
    for intersection in intersection_dictionary.values():
      current_outage_distance = euclidean_distance(outage.latitude, outage.longitude, outage.closest_intersection.latitude, outage.closest_intersection.longitude)
      potential_outage_distance = euclidean_distance(outage.latitude, outage.longitude, intersection.latitude, intersection.longitude)
      if potential_outage_distance < current_outage_distance:
        outage.closest_intersection = intersection
    i += 1
    writer.writerow([outage.id, outage.latitude, outage.longitude, outage.closest_intersection.id, outage.closest_intersection.key])

intersection_dictionary = get_routing_data()
find_closest_intersections(intersection_dictionary)