#lowhighfieldreg
Spatiotemporal image registration allows for comparison of multiple samples from a subject at different time points. Analysing the morphological changes of anatomical structures over a period of time is important in many clinical studies, this comparison necessitates registering the samples/images to a common coordinate space.  Such analysis can be performed by registering the acquired images at different time points. For a typical longitudinal studies, multiple high resolution images at different time points need to be acquired for each subject. While the result from these high-resolution images can be considered "gold-standard" as they maximimize signal to resolution ratio, the acquisition process is less accessible, not sustainable (with high carbon emmission) and more expensive. However, for longitudinal analayis high field images that vary over time is necessary to be able to track the changes, this is a limiting factor as acqusition of multiple high field images per subject is a time consuming and computationally expensive process. We propose a method that infers the local changes in shape using noisy/low field data. We implement a ..... that.....




Spatiotemporal image registration allows for comparison of multiple samples from a subject at different time points. Analysing the morphological changes of anatomical structures over a period of time is important in many clinical studies, this comparison necessitates registering the samples/images acquired at different time points to a common coordinate space. For a typical longitudinal studies, multiple high resolution images at different time points need to be acquired for each subject. While the result from these high-resolution images can be considered "gold-standard" as they try to maximimize signal to noise ratio, the acquisition process is less accessible, not sustainable and more expensive. We propose a method that elminates the need for high resolution images at each time point, our method uses a \emph{single} high resolution image at time t_i and low field images at other time points. We implement an INR that maps the set of coordinate points to a "latent high field" representation of the low field images, we generate a low field representation of these latents which we then use to minimize the loss.


However, for longitudinal analayis high field images that vary over time is necessary to be able to track the changes, this is a limiting factor as acqusition of multiple high field images per subject is a time consuming and computationally expensive process. We propose a method that infers the local changes in shape using noisy/low field data. We implement a ..... that.....




*to do*
1. write abstract 
2. Draw figure
3. use just single paired images(one high res another low res), register both images using INR and 
4. define INR such that it takes \t_o 





We describe what we think the shape of the brain will look like at any point in time given the baseline image!!