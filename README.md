# Computer Vision 

**Why?**
I primarily work with opencv for most of my computer vision tasks and it's documentation sucks!!. So I've decided to write small documentation + explanation.

## Image Pre-Processing Techniques
1. Simple Image Pre-Processing Techniques
    (Rotation, Warping, Translation)
2. Image Thresholding 
3. Connected Components and Contour Detection
4. Edge Detection Algorithms
5. Histogram Equalization
6. Filtering(Linear Filters and Non-Linear Filters)
7. Frequency Domain Analysis
8. Edge Detection
9. Image Similarity Detection
10. Feature Transformation(SIFT, SURF, AKAZE etc.)
11. Feature Matching


**Libraries used:**

1. ```opencv```
2. ```numpy```
3. ```matplotlib```

*Most of the calculations will be done in numpy.*

## Basics Of Images
Images are just a 3-D matrix of 8 bit numbers: 0-255 (Will be referred as ```uint8```). Where 0 means black and 255 means white. Images are stored in an array:

```(height, width, dimensions)```

dimensions = 3 for colored images
dimensions = 1 for grayscale images


## Simple Image Pre-Processing.
One of the most important techniques and probably overlooked in Computer Vision are the most basic ones. 

Techniques:

1. **Rotation**

    For a given iamge, you rotate the image by a certain angle $\theta$. How rotation works in OpenCV could be a little different from the rotation that we've learned. The operation is same but is more generalized in OpenCV.

    ```python
    def rotate_image(img:np.ndarray, angle:int) -> np.ndarray:
    """
    Rotate the image by theta degrees.
    """
        rows, cols = img.shape
        M = cv2.getRotationMatrix2D(((cols-1)/2.0(rows-1)/2.0), angle, 1)
        img_rotated =  cv2.warpAffine(img, M, (cols, rows)) 
        print(img_rotated.shape)
        return img_rotated
    ```
    For more Information, Refer to rotation.ipynb.


## Image Thresholding:

An image basically has 3 channels: Red, Green, Blue will be further referred as **RGB**. For some reason OpenCV uses **BGR**. (Images have another channel, alpha channel but we'll ignore it). In any image pre-processing, if we can decrease dimensions without compromising the information, we tend to do so. (*Google Curse of Dimensionality*). An image can be converted into grayscale by taking mean over all dimensions but we could go one step further; binarizing the image. This technique is known as image thresholding. 

**Algorithm:** For any pixel 'p' in an image matrix M and a given threshold 't'.
            
            if p < t:
                p = 0
            else:
                p = 255

The algorithm is fundamentally simple but there's a catch. How do we find the optimum threshold?

In general, there are two kinds of thresholding techniques. 

1. Global Thresholding
2. Local Thresholding

**Global Thresholding**

In Global Thresholding, there is a single threshold for an entire image Matrix. Based on the given image, a single threshold is calculated. 
Some of the methods of Global Thresholding:

1. **Otsu Thresholding**
2. **Entropy Based Method**
3. **Based on Histogram Analysis**

**Local Thresholding**

A given image Matrix is sub-divided into smaller matrices(sub-images). For each sub-image a threshold is computed and threshold is applied. 
Some of the methods for Local Thresholding:

1. **Niblack's Binarization Method**
2. **Adaptive Thresholding**
3. **PASS Algorithm**

#TODO: Explain all algorithms with citations and code.


## Connected Components and Contour Detection




