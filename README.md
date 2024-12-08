# Image_Segmentation
Developed app using open-source models that segment objects and predict depth estimation within an image.  

**Segmentation:**
1. Take an input image and user-defined points.
2. Use the SAM model (Segment Anything Model) to generate segmentation masks for objects near those points.
3. Overlay those masks on the original image using a red color with transparency.
4. Display the final segmented image.
