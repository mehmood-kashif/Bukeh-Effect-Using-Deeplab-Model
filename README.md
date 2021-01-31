# Bukeh Effect Using Deeplab Model

## Bokeh Effect

 Bokeh is the shallow depth of field effect which blurs the background of portrait photos (typically). It is used to bring emphasis to the subject in the       foreground.

## Semantic Segmentation

Semantic segmentation is defined as assigning class to each pixel in a given image. The main task of segmentation is simplifying or changing the representation of image into segments which are easier to analyse and more meaningful. Semantic segmentation is different from the classification in a way that classification assigns class to the whole image whereas semantic segmentation classifies each image pixel to one of the class.

## Pipeline


![pipeleine](https://user-images.githubusercontent.com/69388951/106387069-d1a21900-63f9-11eb-9d80-939d1063b362.JPG)



## Deeplab V3

We have used Deeplab V3 model for semantic segmentation. The code automatically download the model. 





![model](https://user-images.githubusercontent.com/69388951/106385154-8afbf100-63f0-11eb-8fe6-b5de73bd331c.png)





## Morphological operations

I have used some morphological operations to remove noise from the image and to sharpen the image edges. I applied dilation and erosion operations on the binary mask to enhance the predicted results.
Some predicted Results

### Input Image



![original](https://user-images.githubusercontent.com/69388951/106396772-e1d2ec00-642b-11eb-9355-9873dc50f04c.png)





### Binary Mask

![mask](https://user-images.githubusercontent.com/69388951/106396849-5b6ada00-642c-11eb-84c9-390160d55048.png)








### Blur Image





![blur](https://user-images.githubusercontent.com/69388951/106396866-63c31500-642c-11eb-9cd6-b27e35a71c27.png)






### Bukeh Effect




![bukeh](https://user-images.githubusercontent.com/69388951/106396873-6aea2300-642c-11eb-990f-2b786e4b225c.jpg)

## For inference 


python3 inference.py -input images

