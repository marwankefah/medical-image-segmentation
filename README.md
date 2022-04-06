# medical-image-segmentation
segmentation models pytorch with MONAI transforms in  medical imaging

FETA dataset with 20% labeled data | Dice Score
---------------------------------|------------------------
Mean Teacher ResNet-34          |  **0.668 ± 0.21**
Supervised Approach ResNet-34|  0.634 ± 0.20


Segementation models pytorch with Medical Imaging Data Augmentation (MONAI) 


Original Image           |  Elastic Deformation  
:-------------------------:|:-------------------------:
![](https://github.com/marwankefah/medical-image-segmentation/blob/master/imgs_readme/original.png)  |  ![](https://github.com/marwankefah/medical-image-segmentation/blob/master/imgs_readme/randdeform.png)
 
 Random Affine        |  Random Gaussian Noise
:-------------------------:|:-------------------------:
![](https://github.com/marwankefah/medical-image-segmentation/blob/master/imgs_readme/randaffine.png)  |  ![](https://github.com/marwankefah/medical-image-segmentation/blob/master/imgs_readme/randGaussian.png)

  Random Flip         |  Random Blur
:-------------------------:|:-------------------------:
![](https://github.com/marwankefah/medical-image-segmentation/blob/master/imgs_readme/flip1.png)  |  ![](https://github.com/marwankefah/medical-image-segmentation/blob/master/imgs_readme/randsmooth.png)

  Random Rotate         |  Random Zoom
:-------------------------:|:-------------------------:
![](https://github.com/marwankefah/medical-image-segmentation/blob/master/imgs_readme/randrot.png)  |  ![](https://github.com/marwankefah/medical-image-segmentation/blob/master/imgs_readme/randzoom.png)




Bano, S. et al. (2020). Deep Placental Vessel Segmentation for Fetoscopic Mosaicking. In: , et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2020. MICCAI 2020. Lecture Notes in Computer Science(), vol 12263. Springer, Cham. https://doi.org/10.1007/978-3-030-59716-0_73

