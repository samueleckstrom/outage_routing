import csv
import statistics

def get_stats(number_of_centroids, number_of_cars):
  staticcasemaximums = []
  staticcasemedians = []
  staticcasemeans = []
  basecasemaximums = []
  basecasemedians = []
  basecasemeans = []
  maximumdifferences = []
  mediandifferences = []
  meandifferences = []
  while number_of_centroids < 15:
    try:
      f = open('./staticcase-2020-12-09to11/routing_data_centroids' + str(number_of_centroids) + '_cars' + str(number_of_cars) + '.csv', 'r')
      staticcasecsvreader = csv.reader(f, delimiter=',')
      staticcasenumbers = []
      for row in staticcasecsvreader:
        if row[6] == 'Maximum Vehicle Repair Time':
          continue
        elif row[6] != '':
          staticcasenumbers.append(float(row[6]))
        else:
          staticcasenumbers.append(0.0)
      staticcasemaximums.append(max(staticcasenumbers))
      staticcasemedians.append(statistics.median(staticcasenumbers))
      staticcasemeans.append(statistics.mean(staticcasenumbers))
      f = open('./basecase-2020-12-09to11/routing_data_centroids' + str(number_of_centroids) + '_cars' + str(number_of_cars) + '.csv', 'r')
      basecasecsvreader = csv.reader(f, delimiter=',')
      basecasenumbers = []
      for row in basecasecsvreader:
        if row[6] == 'Maximum Vehicle Repair Time':
          continue
        elif row[6] != '':
          basecasenumbers.append(float(row[6]))
        else:
          basecasenumbers.append(0.0)
        basecasemaximums.append(max(basecasenumbers))
        basecasemedians.append(statistics.median(basecasenumbers))
        basecasemeans.append(statistics.mean(basecasenumbers))
      maximumdifferences.append([b - s for s, b in zip(staticcasemaximums, basecasemaximums)])
      mediandifferences.append([b - s for s, b in zip(staticcasemedians, basecasemedians)])
      meandifferences.append([b - s for s, b in zip(staticcasemeans, basecasemeans)])

      if number_of_cars > number_of_centroids * 4:
        number_of_centroids += 1
        number_of_cars = number_of_centroids * 2
      else:
        number_of_cars += number_of_centroids
    except:
      if number_of_cars > number_of_centroids * 4:
        number_of_centroids += 1
        number_of_cars = number_of_centroids * 2
      else:
        number_of_cars += number_of_centroids
      print('skipping')
  maximumdifferences = [j for sub in maximumdifferences for j in sub]
  mediandifferences = [j for sub in mediandifferences for j in sub]
  meandifferences = [j for sub in meandifferences for j in sub]
  print(statistics.mean(maximumdifferences), statistics.stdev(maximumdifferences))
  print(statistics.mean(mediandifferences), statistics.stdev(mediandifferences))
  print(statistics.mean(meandifferences), statistics.stdev(meandifferences))


number_of_cars = 10
number_of_centroids = 5
get_stats(number_of_centroids, number_of_cars)