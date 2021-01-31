Bukeh Effect Using Deeplab Model

 Bokeh Effect

 Bokeh is the shallow depth of field effect which blurs the background of portrait photos (typically). It is used to bring emphasis to the subject in the       foreground.

Semantic Segmentation

Semantic segmentation is defined as assigning class to each pixel in a given image. The main task of segmentation is simplifying or changing the representation of image into segments which are easier to analyse and more meaningful. Semantic segmentation is different from the classification in a way that classification assigns class to the whole image whereas semantic segmentation classifies each image pixel to one of the class.

Model

We have used Deeplab V3 model for semantic segmentation. The code automatically download the model. 

Deeplab V3





Morphological operations

I have used some morphological operations to remove noise from the image and to sharpen the image edges. I applied dilation and erosion operations on the binary mask to enhance the predicted results.
Some predicted Results

Input Image

Binary Mask
