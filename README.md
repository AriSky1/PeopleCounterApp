


The challenges:
-very small objects, too small for HOG (Histograms of Oriented Gradients),trained to detect pedestrians that are mostly standing up
and fully visible, so its performance may be limited in other cases.
- objects in motion on a live stream require very fast processing


Choice : Background substractor method with MOG2 (OpenCV)
Real-time human detection system based on motion analysis


limits:
-works better on same size objects
- sensitive to car flashes, rain, night lights
- hard to parameter
- works only under certain conditions

