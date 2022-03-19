# Computer Vision 

**Why?**
I primarily work with opencv for most of my computer vision tasks and it's documentation sucks!!. So I've decided to write small documentation + explanation.

## Image Pre-Processing Techniques
1. Simple Image Pre-Processing Techniques
    (Rotation, Warping, Thresholding)
2. Connected Components and Contour Detection
3. Histogram Equalization
4. Filtering(Linear Filters and Non-Linear Filters)
5. Frequency Domain Analysis
6. Edge Detection
7. Image Similarity Detection
8. Feature Transformation(SIFT, SURF, AKAZE etc.)


**Libraries used:**

1. ```opencv```
2. ```numpy```
3. ```matplotlib```

*Most of the calculations will be done in numpy.*

### Simple Image Pre-Processing.
One of the most important techniques and probably overlooked in Computer Vision are the most basic ones. 

Techniques:

1. Rotation
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
    

    



