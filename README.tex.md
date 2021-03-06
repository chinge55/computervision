# Computer Vision 

**Disclaimer:** These algorithms will look pretty dumb until you get the thrill of it and see for yourself what magic you can do with them.

## Image Pre-Processing Techniques
1. Simple Image Pre-Processing Techniques
    (Rotation, Morphological Operations)
2. Image Thresholding 
3. Morphological Operations
4. Connected Components and Contour Detection
5. Edge Detection Algorithms
6. Histogram Equalization
7. Filtering(Linear Filters and Non-Linear Filters)
8. Frequency Domain Analysis
9. Image Similarity Detection
10. Feature Extraction(SIFT, SURF, AKAZE etc.)
11. Template Matching
12. Anisotropic Filters
13. Image Moments

## Neural Networks
1. Training Common Mistakes


**Libraries used:**

1. ```opencv```
2. ```numpy```
3. ```matplotlib```


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


## Morphological Operations
(*Morphological Operations done only on binarized Images*)
Code: morphology.ipynb

They are non-linear operations related to shape and morphology of an image. 
If we're doing morphological operations on images then we are going to need an image kernel. A 2D matrix smaller than the image.

(Definition could look intimidating but they are one of the most easy to understand algorithms in Image Processing)

1. Image Dilation:
    
    For a white pixel inside a binarized image, convert it's neighboring black pixels to white. <br>
    Eg. If I have a white circle of radius r. It'll have a bigger circle or radius r' after dilation. 
    
    For two given matrices, an image matrix and a structuring matrix. The structuring matrix is superimposed into the image matrix. A pixel element is 1(white) if at least one pixel under the kernel (after superimposition) is 1(white)
2. Image Erosion:
    
    Opposite of dilation 

    The structuring matrix is superimposed into the image matrix. A pixel element is white if all the pixels under the kernel (after superimposition) are 1 (white)

    
3. Image Opening:

    Image Opening is Erosion Followed by Dilation.
4. Image Closing

    Image closing is Dilation Followed by Erosion.

    *Personally, I don't ever think about Opening or closing Operation. Eg. If I have small white noises around a big white blob laid on an empty canvas. What I'd like to do is remove the smaller blocks. So, I'd first erode the image so that the smaller white blocks would be removed and not the main blob. After that, I would dilate the image so that the original blob of interest would come back to its original size(nearly) And sometimes, you'd want to use different kernels for opening and closing too.*
5. Image Skeletonization

    Skeletonization is the process of reducing foreground regions of a binary image to a skeleton-like image preserving the connectivity of the original region. (Well, Basically creating a skeleton). Rather than explaining, it's better to view the code and dissect it to view how it works and I leave it up to the reader to refer the code and do so. 

## Connected Components and Contour Detection
*Code: connected_components_contour.ipynb*

### Connected Components:

Connected Component Labelling is used in Computer Vision to detect regions in binary digital images, although color images with higher dimensionality can also be processed. 

Connected Components Algorithm is one of the fundamentally simpler algorithms. For any binarized image, find all the pixels(white) that are connected to each other and label them. 
Eg. 
$$
\begin{bmatrix}
0&0&0&0&0&0&0&0&0&0&0&0&0&0\\
0&0&255&255&0&0&0&0&0&0&0&0&0&0\\
0&255&0&255&255&0&0&0&0&0&0&0&0&0\\
0&0&255&0&0&0&0&0&0&0&0&0&0&0\\
0&0&255&0&0&0&0&0&0&255&0&0&0&0\\
0&0&255&0&0&0&0&0&0&255&0&0&0&0\\
0&0&255&0&0&0&0&0&0&255&255&255&0&0\\
0&0&255&0&0&0&0&0&255&255&255&255&0&0\\
0&0&0&0&0&0&0&0&0&0&0&0&0&0\\
0&0&0&0&0&0&0&0&0&0&0&0&0&0\\
\end{bmatrix}
$$ 

As we can see tht there are two distinct white patches (255) inside the image. What connected component labelling does is to label those two patches based on whether they are connected or not. 
$$
\begin{bmatrix}
0&0&0&0&0&0&0&0&0&0&0&0&0&0\\
0&0&1&1&0&0&0&0&0&0&0&0&0&0\\
0&1&0&1&1&0&0&0&0&0&0&0&0&0\\
0&0&1&0&0&0&0&0&0&0&0&0&0&0\\
0&0&1&0&0&0&0&0&0&2&0&0&0&0\\
0&0&1&0&0&0&0&0&0&2&0&0&0&0\\
0&0&1&0&0&0&0&0&0&2&2&2&0&0\\
0&0&1&0&0&0&0&0&2&2&2&2&0&0\\
0&0&0&0&0&0&0&0&0&0&0&0&0&0\\
0&0&0&0&0&0&0&0&0&0&0&0&0&0\\
\end{bmatrix}
$$ 

The second image matrix is the matrix gotten after connected component labelling. I think what it does is pretty self-explanatory. 

Algorithm: 

(*There have been various algorithms on connected components, but I'll write a simple and inefficient method*)

1. For pixel in img row, cols
2. component_no = 0
3. If the pixel is 0, continue
4. For the pixel, check it's neighbors
    if neighbor is already labelled, 
        
        if neighbors have multiple labels:
            change it's label to the neighbor's label (lower).
            change all the pixels of other label to the lower label
        
        (If, one pixel has two neighbors 1 and 2. Change the pixel to 1. And change all the pixels of 2 to 1 as this pixel connects two blobs.)

    Else

        pixel = component_no; component_no += 1

### Contour Detection:

We could take the base as connected components and try to add a boundary following algorithm. What we're trying to get is a proper boundary of all the connected components, and in turn detect contours for a given image. 



## Image Moments

You might have heard about moments in statistics:
$$
    \mu_n = \int_{-\infty}^{+\infty} (x-c)^n f(x)dx
$$

In simple terms, image moments are a set of statistical parameters to measure the distribution of where the pixels are and their intensities. 

Mathematically, the image moment $M_{ij}$ of order $(i,j)$ for a grayscale image with pixel intensities $I(x,y)$ is calculated as

$$
M_{ij} = \sum_{x}\sum_{y}x^iy^jI(x,y)
$$
Where, x, y refers to the row and column index and $I(x,y)$ refers to the intensity at that location $(x,y)$.
(From the moment equation, replace integration by summation, well images are discrete. And as images have two coordinates, change the equation to a multi-variate equation)

**Simple Uses Of Image Moments:**
(Used to describe properties of a binary image)
1. Calculating Area: (Zeroth Order Moment)

    To calculate the area of a binary image, you'd need to calculate the first moment. 
    $$
    M_{0,0} = \sum_{x = 0}^{w} \sum_{y = 0}^h x^0y^0 f(x,y)
    $$
    As, $x^0$ and $y^0$ don't have any effect, can be removed
    $$
    M_{0,0} = \sum_{x = 0}^{w} \sum_{y = 0}^h f(x,y)
    $$

    This, might look intimidating but converting it to code might change your perspective.
    ```python
    def get_area(img):
        height, width = img.shape
        area = 0
        for w in range(0, width):
            for h in range(0, height):
                area += img[h,w]
        return area
    # Easier and faster method
    area = np.sum(img)
    #Or
    area = cv2.moments(img)['m00']
    ```

    #TODO Run this code (simplest codes might not run sometimes)
    

2. Calculating Centroid: (First Order Moment)
    
    Centroid of an image is just a pixel location. Which is given by:
    $$
    centroid = (\frac{\mu_{1,0}}{\mu_{0,0}}, \frac{\mu_{0,1}}{\mu_{0,0}})
    $$
    ```python
    def get_centroid(img):
        mu = cv2.moments(img)
        centroid = mu['m10']//mu['m00'], mu['m01']//mu['m00']
        return centroid
    ```
