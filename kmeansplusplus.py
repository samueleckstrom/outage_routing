from classes import Outage, Centroid, Cluster, CentroidDifference
import distances
from random import randint, random, sample

def k_means_plus_plus(outages, number_of_centroids, number_of_iterations):
  starting_centroid = outages[randint(0, len(outages) - 1)]
  centroids = [Centroid(0, starting_centroid.latitude, starting_centroid.longitude)]
  for i in range(number_of_centroids - 1):
    distance_to_closest_centroids = []
    total_distance = 0.0
    for outage in outages:
      (_, _, _, distance) = distances.closest_centroid(outage.latitude, outage.longitude, centroids)
      total_distance += distance
      distance_to_closest_centroids.append([outage, total_distance])
    random_probability = total_distance * random()
    j = 0
    while True:
      if distance_to_closest_centroids[j][1] > random_probability:
        break
      j += 1
    centroids.append(Centroid(i + 1, distance_to_closest_centroids[j][0].latitude, distance_to_closest_centroids[j][0].longitude))    

  for i in range(number_of_iterations):
    k_means_differences = [CentroidDifference() for j in range(len(centroids))]
    for outage in outages:
      (key, latitude_difference, longitude_difference, _) = distances.closest_centroid(outage.latitude, outage.longitude, centroids)
      k_means_differences[key].latitude_difference += latitude_difference
      k_means_differences[key].longitude_difference += longitude_difference
      k_means_differences[key].number_of_differences += 1

    for centroid in centroids:
      centroid.latitude += (k_means_differences[centroid.key].latitude_difference / k_means_differences[centroid.key].number_of_differences) if k_means_differences[centroid.key].number_of_differences != 0 else 0.0
      centroid.longitude += (k_means_differences[centroid.key].longitude_difference / k_means_differences[centroid.key].number_of_differences) if k_means_differences[centroid.key].number_of_differences != 0 else 0.0
    
  clusters = { }
  for outage in outages:
    (key, _, _, _) = distances.closest_centroid(outage.latitude, outage.longitude, centroids)
    if key in clusters:
      clusters[key].outages.append(outage)
    else:
      clusters[key] = Cluster(centroids[key], outage)
  return clusters