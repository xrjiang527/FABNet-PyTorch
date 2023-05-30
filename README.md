FABNet-PyTorch
------------------------------------------------------------------------------------------------------------------------------------------
This is a pytorch project for the paper FABNet: Frequency-Aware Binarized Network for Single Image Super-Resolution by Xinrui Jiang, Nannan Wang, Jingwei Xin, Keyu Li, Xi Yang, Jie Li, Xiaoyu Wang and Xinbo Gao

Introduction
------------------------------------------------------------------------------------------------------------------------------------------
This paper presents a frequency-aware binarized network (FABNet) for single image super-resolution. We decompose the image features into low-frequency and high frequency components and then adopt a "divide-and-conquer" strategy to process them with well-designed binary network structures.
![image](https://user-images.githubusercontent.com/54347263/236616873-8cfd7271-9619-434e-bbcb-6ae3fa3e4871.png)

Dependencies and Installation
------------------------------------------------------------------------------------------------------------------------------------------
conda create -n FABNet python=3.7<br>
conda activate FABNet<br>
conda install pytorch=1.2 torchvision=0.4 cudatoolkit=10.0 -c pytorch<br>
pip install pytorch-wavelets matplotlib scikit-image opencv-python h5py tqdm<br>

Dataset
------------------------------------------------------------------------------------------------------------------------------------------
You can use the following links to download the datasets:<br>
DIV2K dataset https://cv.snu.ac.kr/research/EDSR/DIV2K.tar<br>
Set5 http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html<br>
Set14 https://sites.google.com/site/romanzeyde/research-interests<br>
BSD100 https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/<br>
Urban100 https://sites.google.com/site/jbhuang0604/publications/struct_sr<br>

Pretrained Model
------------------------------------------------------------------------------------------------------------------------------------------
You can download pre-trained models from: https://pan.baidu.com/s/13SDor2gkPMCmWlp7PZ592A (code：i49h)
You can download results from: https://pan.baidu.com/s/1xYEUcNV2_ZY486NhCRSNkg (code：5jpy) 

Usage
------------------------------------------------------------------------------------------------------------------------------------------
Train<br>
Train the model on the corresponding network using the train config. 
For example, the training on FABNetC12B4:<br>
train_model:  import model.FABNet_tiny as model<br>
python train.py --n_feats 12<br>
The training on FABNetC96B8:<br>
train_model:  import model.FABNet_large as model<br>
python train.py --n_feats 96<br>
Test<br>
Test the model on the corresponding network using the test config with the pretrained model. 
For example, the testing on FABNetC12B4:<br>
train_model:  import model.FABNet_tiny as model<br>
python test.py --data_type img --n_feats 12 --pre_model */FABNet_B4C12_X2.pth <br>


