import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import savefig
from scipy.stats import gaussian_kde
import smopy

# Get the data
df  = pd.read_csv('airports.csv')
lon = df['Longitude']
lat = df['Latitude']

####################
### Discrete KDE ###
####################

# Calculate the point density
xy = np.vstack([lon,lat])
density = gaussian_kde(xy)(xy)

# Create a map
map = smopy.Map((min(lat), min(lon), max(lat), max(lon)))
x, y = map.to_pixels(lat, lon)
ax = map.show_mpl(figsize=(16, 12))
ax.scatter(x, y, c = density, s = 4, edgecolor = '')
plt.show()
#savefig('Airports.pdf')

##################
### Smooth KDE ###
##################

import seaborn as sbn
data = list(zip(xy[0], xy[1]))
data = pd.DataFrame(data, columns=('x', 'y'))
sbn.kdeplot(data.x, data.y, shade=True)
plt.show()





