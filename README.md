


https://user-images.githubusercontent.com/105871709/227682244-9def5fe6-7a77-4ab7-99c6-931738e08350.mp4



The challenges:
<br>
- very small objects, too small for HOG (Histograms of Oriented Gradients),trained to detect pedestrians that are mostly standing up
and fully visible
- objects in motion on a live stream require very fast processing


Choice : Background substractor method with MOG2 (OpenCV)


limits:
<br>
- works better on same size objects
- sensitive to car flashes, rain, night lights
- hard to parameter
- works only under certain conditions

