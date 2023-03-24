
Background substractor method with MOG2 (Moving Oriented Gradients)
Real-time human detection system based on motion analysis

HOG method
HOG (Histograms of Oriented Gradients)  method, trained to detect pedestrians that are mostly standing up
and fully visible, so its performance may be limited in other cases.


In computer vision and image processing, createBackgroundSubtractorMOG2 is a function provided by the OpenCV library,
which creates an instance of the BackgroundSubtractorMOG2 class.
This class is used to perform foreground/background segmentation,
which is a common preprocessing step in many computer vision tasks.

The "MOG" in createBackgroundSubtractorMOG2 stands for "Mixture of Gaussians" .
The BackgroundSubtractorMOG2 class implements the Gaussian mixture model (GMM) based background subtraction algorithm
 described in the papers by Zivkovic  and Stauffer and Grimson.
 The algorithm uses a mixture of Gaussian distributions to model the background of a scene
  and then detects foreground regions as pixels that do not fit this model.

The createBackgroundSubtractorMOG2 function takes four parameters:
history, nmixtures, backgroundRatio, and noiseSigma [3].
These parameters control the size of the history window, the number of Gaussian mixtures
 used to model the background, the ratio between the foreground and the background,
 and the strength of the noise in the input images.