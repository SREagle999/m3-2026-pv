import geopandas as gpd
import matplotlib.pyplot as plt

gdf = gpd.read_file("datasets/nashville_zip_map/Zip_Codes_Boundaries.shp")
gdf = gdf.sort_values(by='ZipCode')
# print(gdf['ZipCode'])
# print(gdf.columns)

# example heatmap - would use previously computed data
gdf['Vulnerability'] = 0.0
for i in range(0,len(gdf['ZipCode'])):
    gdf.loc[i, 'Vulnerability'] = i / (len(gdf) - 1)

fig, ax = plt.subplots()
gdf.plot(column='Vulnerability', cmap='Reds', ax=ax, edgecolor='black') # choose column

ax.set_title('Nashville Zip Codes')
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
plt.show()