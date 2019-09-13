# Deep  Hierarchical  Encoder-Decoder  with  Point Atrous  Convolution  for  Unorganized  3D  Points

![prediction example](https://github.com/paul007pl/PointAtrousGraph/blob/master/misc/point_atrous_conv_figure.png)


### Introduction
This work is based on our [arXiv tech report](https://arxiv.org/abs/1907.09798).

We propose a deep hierarchical Encoder-Decoder architecture with point atrous convolution to exploit multi-scale edge-aware features in unorganized 3D points.

Experimental results show that our network outperform previous state-of-the-art methods, in 3D object classification, object-part segmentation and semantic segmentation.
In particular, our proposed modules are more efficient (in terms of required training time and memory footprint) than previous networks which heavily rely on neighboring points.

We encourage you to apply our proposed modules for more complicated point cloud applications.


### Installation
The code has been tested with Tensorflow 1.4, CUDA 8.0 and Tensorflow 1.12, CUDA 9.0

1. install required python libs
2. download correspoinding dataset
3. compile all the tensorflow ops


### Citation
If you find our work useful in your research, please consider citing:

	@article{pan2019pointatrousnet,
	  title={PointAtrousNet: Point Atrous Convolution for Point Cloud Analysis},
	  author={Pan, Liang and Wang, Pengfei and Chew, Chee-Meng},
	  journal={IEEE Robotics and Automation Letters},
	  volume={4},
	  number={4},
	  pages={4035--4041},
	  year={2019},
	  publisher={IEEE}
	}

	@article{pan2019pointatrousgraph,
	  title={PointAtrousGraph: Deep Hierarchical Encoder-Decoder with Atrous Convolution for Point Clouds},
	  author={Pan, Liang and Chew, Chee-Meng and Lee, Gim Hee},
	  journal={arXiv preprint arXiv:1907.09798},
	  year={2019}
	}


Our work is inspired by previous work: PointNet, PointNet++ and DGCNN.
If you apply their modules, please consider citing their papers also:
	
	@article{qi2016pointnet,
	  title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
	  author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
	  journal={arXiv preprint arXiv:1612.00593},
	  year={2016}
	}

	@inproceedings{qi2017pointnet++,
	  title={Pointnet++: Deep hierarchical feature learning on point sets in a metric space},
	  author={Qi, Charles Ruizhongtai and Yi, Li and Su, Hao and Guibas, Leonidas J},
	  booktitle={Advances in neural information processing systems},
	  pages={5099--5108},
	  year={2017}
	}

	@article{dgcnn,
	  title={Dynamic Graph CNN for Learning on Point Clouds},
	  author={Wang, Yue and Sun, Yongbin and Liu, Ziwei and Sarma, Sanjay E. and Bronstein, Michael M. and Solomon, Justin M.},
	  journal={ACM Transactions on Graphics (TOG)},
	  year={2019}
	}

### License
Our code is released under MIT License.


