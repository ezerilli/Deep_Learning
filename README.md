# Deep Learning
https://github.com/ezerilli/Deep_Learning

### SETTING UP THE ENVIRONMENT 👨🏻‍💻👨🏻‍💻👨🏻‍💻

The following steps lead to setup the working environment for Deep Learning projects. 👨🏻‍💻‍📚‍‍‍‍

Installing the conda environment is a ready-to-use solution to be able to run python scripts without having to worry 
about the packages and versions used. Alternatively, you can install each of the packages in `requirements.yml` on your 
own independently with pip or conda.

1. Start by installing Conda for your operating system following the instructions [here](https://conda.io/docs/user-guide/install/index.html).

2. Now install the environment described in `requirements.yaml`:
```bash
conda env create -f requirements.yml
```

4. To activate the environment run:
```bash
conda activate DL
```

5. Once inside the environment, if you want to run a python file, run:
```bash
python my_file.py
```

6. To deactivate the environment run:
```bash
conda deactivate
```

7. During the semester I may need to add some new packages to the environment. So, to update it run:
```bash
conda env update -f requirements.yml
```

### Fashion-MNIST ‍🔥🔥🔥

This assignment aims to explore the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) and its 
applications to clothing detection and recognition in real world images and videos.

The project consists of two parts: 

- _experiment 1_, training and testing 3 different models (MiniVGG4, MiniVGG6 and MiniVGG6 with Data Augmentation ) and 
producing corresponding accuracy/loss curves.

- _experiment 2_, clothing detection and recognition in real world images and videos (work in progress...).

In order to run the experiments, run:
```bash
cd Fashion_MNIST
python run_experiments.py
```
Figures will show up progressively. It takes a while to perform all the experiments. By default, models training is commented 
out and the script loads pre-trained weights and then assesses performances on the test set. However, training plots have 
already been saved into the images directory. Theory, results and experiments are discussed in the report.

### REFERENCES

- [1] Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. *Gradient based learning applied to document recognition*. IEEE, 1998.
- [2] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. *ImageNet Large Scale Visual Recognition Challenge*. International Journal of Computer Vision (IJCV), 2015.
- [3] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. *Imagenet Classification with Deep Convolutional Neural Networks*. NIPS, 2012.
- [4] Karen Simonyan and Andrew Zisserman. *Very Deep Convolutional Networks for Large-Scale Image Recognition*. arXiv preprint arXiv:1312.6082v4, 2014.
- [5] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. *Going Deeper with Convolutions*. In Computer Vision and Pattern Recognition (CVPR), 2015.
- [6] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. *Deep Residual Learning for image recognition*. arXiv preprint arXiv:1512.03385, 2015.
- [9] Han Xiao, Kashif Rasul, and Roland Vollgraf. *Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms*, 2017.
- [10] Keras: The Python Deep Learning library. https://keras.io. Last accessed: 2019-10-24.
- [11] Google Cloud Platform for fast Deep Learning development. https://cloud.google.com/deep-learning-vm/. Last accessed: 2019-10-24.