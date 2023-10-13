# Fine-grained Hand Bone Segmentation via Adaptive Multi-dimensional Convolutional Network and Anatomy-constraint Loss

This is a reference implementation of our paper in MICCAI2023:
Fine-grained Hand Bone Segmentation via Adaptive Multi-dimensional Convolutional Network and Anatomy-constraint Loss.
[Paper](https://link.springer.com/chapter/10.1007/978-3-031-43901-8_38)

Ultrasound imaging is a promising tool for clinical hand examination due to its radiation-free and cost-effective nature.
To mitigate the impact of ultrasonic imaging defects on accurate clinical diagnosis, automatic fine-grained hand bone segmentation 
is highly desired. However, existing ultrasound image segmentation methods face difficulties in performing this task due to the 
presence of numerous categories and insignificant inter-class differences. To address these challenges, we propose a novel Adaptive 
Multi-dimensional Convolutional Network (AMCNet) for fine-grained hand bone segmentation. It is capable of dynamically adjusting the 
weights of 2D and 3D convolutional features at different levels via an adaptive multi-dimensional feature fusion mechanism. We also 
design an anatomy-constraint loss to encourage the model to learn anatomical relationships and effectively mine hard samples. 
Experiments demonstrate that our method outperforms other comparison methods and effectively addresses the task of fine-grained hand 
bone segmentation in ultrasound volume. We have developed a user-friendly and extensible module on the 3D Slicer platform based on 
the proposed method and will release it globally to promote greater value in clinical applications.

# Citation
If you find our work is useful for you, please cite us.
