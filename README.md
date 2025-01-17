# Real-time-New-York-City-Traffic-Heatmap
## Introduction
We developed a web application demo, dynamically monitoring and visualizing the traffic road condition change in New York City, especially in Manhattan Island. We made full use of [TomTom](https://developer.tomtom.com/) and [OpenWeather](https://openweathermap.org/api) APIs to request traffic-related streaming data including real-time intersection speed, historical density statistics and weather details. We acquired speed data every 15 minutes, weather data once an hour, density information once a day, and updated the traffic heatmap every 15 minutes. We implemented our visualization by Google Maps APIs and similar to Google Map, we quantified the congestion into three degrees and plot the result by coloring the street with corresponding colors. Our web demo achieved similar performance with Google Maps, but with more reasonable and accurate results. Our work can also accept streaming data to achieve high-level of concurrency.

The following is an introduction to the files in this repository:

`main.py:` Program entry. Everything starts from here.   
`tools/faster_call.py:` Call TomTom Traffic API every 15 minutes and process the returned data.  
`tools/weather_hourly_rdd.py:` Call OpenWeather API every 1 hour and process the returned data.  
`tools/density_do_calculation.py:` The evaluation model for scoring the congestion of roads.  
`tools/traffic_volume_processing.py:` Process traffic density data and output the results.  
`google maps/... :` Front-end visualization components, combined with the use of Google Maps API.  
`downsample/... :` Downsample the street records in Centerline.csv and output the results.  
`fairscheduler.xml:` Spark Scheduler configuration file.  
`Centerline.csv:` Contains more than 140,000 street information in New York City.  
`points.csv:` Contains street coordinates information for Manhattan Island after downsampling.  
`crash_data.csv:` Contains the location and time of the car crash in New York City.  
`points_with_community.csv:` We divided Manhattan Island into 24 regions, and this file contains traffic density information for each region.  `


## Dependency
Python 3.9.10  
Spark 3.2.1  
Flask 2.1.1


## Quick Start

`$ git clone https://github.com/TerryWang11/Real-time-New-York-City-Traffic-Heatmap.git`  
`$ cd Real-time-New-York-City-Traffic-Heatmap`  
`$ Python3 main.py`  
`$ cd google maps/`  
`$ Python3 server.py`  


## License
[MIT](https://github.com/TerryWang11/Real-time-New-York-City-Traffic-Heatmap/blob/main/LICENSE) © Yukun Wang

## Contact
Zhenrui Chen: zc2569@columbia.edu  
Mingzhe Hu:   mh4116@columbia.edu  
Yukun Wang:   yw3536@columbia.edu  
