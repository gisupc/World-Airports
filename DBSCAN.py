import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
import smopy

# Get data
df = pd.read_csv('airports.csv')
coordinates = df.as_matrix(columns = ['Longitude', 'Latitude'])

# Compute DBSCAN
db = DBSCAN(eps=.01, min_samples=2).fit(coordinates)
labels = db.labels_
numClusters = len(set(labels)) - (1 if -1 in labels else 0)
clusters = pd.Series([coordinates[labels == i] for i in range(numClusters)])
print('Number of clusters: %d' % numClusters)

# Compute centroid of an array of points
def getCentroid(points):
    n = points.shape[0]
    sum_lon = np.sum(points[:, 1])
    sum_lat = np.sum(points[:, 0])
    return (sum_lon/n, sum_lat/n)

# Get the nearest point to the centroid
def getNearestPoint(set_of_points, point_of_reference):
    closest_point = None
    closest_dist = None
    for point in set_of_points:
        point = (point[1], point[0])
        dist = great_circle(point_of_reference, point).meters
        if (closest_dist is None) or (dist < closest_dist):
            closest_point = point
            closest_dist = dist
    return closest_point

# Go through every point
lon = []
lat = []
for i, cluster in clusters.iteritems():
    if len(cluster) < 3:
        representative_point = (cluster[0][1], cluster[0][0])
    else:
        representative_point = getNearestPoint(cluster, getCentroid(cluster))
    lon.append(representative_point[1])
    lat.append(representative_point[0])

lon = pd.Series(lon)
lat = pd.Series(lat)

# Make a map
map = smopy.Map((min(lat), min(lon), max(lat), max(lon)))
x, y = map.to_pixels(lat, lon)
ax = map.show_mpl(figsize=(16, 12))
ax.scatter(x, y, s = 30, c='red', edgecolor = '')
plt.show()

