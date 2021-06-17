from classes import Outage, Centroid, Cluster, CentroidDifference
import distances
from random import randint, random, sample

def k_means(outages, number_of_centroids, number_of_iterations):
  starting_centroid_outages = sample(outages, number_of_centroids)
  centroids = [Centroid(intersection_index, starting_centroid_outages[intersection_index].latitude, starting_centroid_outages[intersection_index].longitude) for intersection_index in range(number_of_centroids)]
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