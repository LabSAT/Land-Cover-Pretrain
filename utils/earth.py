import ee

import math 
import requests
import numpy as np 
import matplotlib.pyplot as plt

from io import BytesIO
from PIL import Image



try:
  ee.Initialize()
except Exception as e:
  ee.Authenticate()
  ee.Initialize()



class EasyEarth:
  def __init__(self, dataset):
    self.dataset = ee.ImageCollection(dataset)

  def change_dataset(self, dataset):
    self.dataset = ee.ImageCollection(dataset)

  
  def select_AOI(self, lat, lon, k = 10, dates=None, cloud_name = "CLOUD_COVER" ,cloud_pct=None):
    self.outer_AOI = self.__create_AOI(lat, lon, k / 50)
    self.AOI = self.__create_AOI(lat, lon, k)
    self.data_AOI = self.dataset.filterBounds(self.AOI)
    if dates is not None:
      self.data_AOI = self.__filter_dates(self.data_AOI, dates)
    if cloud_pct is not None:
      self.data_AOI = self.__filter_cloudy(by=cloud_name, cloud_pct=cloud_pct)

    self.AOI_size = self.data_AOI.size().getInfo()

        
  def __create_AOI(self, lat, lon, s=10):
    """Creates a s km x s km square centered on (lat, lon)"""
    v = (180/math.pi)*(500/6378137)*s # roughly 0.045 for s=10
    geometry = ee.Geometry.Polygon([[[lon - v, lat + v],
                                    [lon - v, lat - v],
                                    [lon + v, lat - v],
                                    [lon + v, lat + v]]], None, False)

    return geometry #ee.Geometry.Rectangle([lon - v, lat - v, lon + v, lat + v])

  def __filter_dates(self, col, dates):
    return col.filterDate(dates[0], dates[1])


  def __filter_cloudy(self, by="CLOUDY_PIXEL_PERCENTAGE", cloud_pct=5):
    """
    주로 Sentinel(COPERNIQUS)에서 구름 양을 조절하기 위해 사용합니다.
    """
    return self.data_AOI.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_pct))



  def sort_by(self, method = 'CLOUD_COVER', ascending=True):
    sorted_collections = self.data_AOI.sort(method, ascending)
    self.data_AOI = sorted_collections


  def plot_image(self, img, paramters, cloud_name = 'CLOUD_COVER'):
    
    cloud_pct = img.get(cloud_name).getInfo()
    # print(f'Cloud Cover (%): {cloud_pct}')

    url = img.getThumbUrl(parameters)
    response = requests.get(url)
    
    return cloud_pct, Image.open(BytesIO(response.content))


  def get_collections_at(self, idx, collections=None):
    if collections is not None:
      size = collections.size().getInfo()
      return ee.Image(collections.toList(size).get(idx))
    else:
      return ee.Image(self.data_AOI.toList(self.AOI_size).get(idx))

  def save_image(self, save_path, image):
    plt.imsave(save_path, np.array(image))
    # print(str(save_path) + " complete")

  def calculate_alpha_ratio(self, image):
    img_array = np.array(image)
    return np.sum(img_array == 0) / img_array.size