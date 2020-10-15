# Fashion Category and Attribute Prediction!

The goal of this project is to predict the categories and attributes of the clothes. There are 46 categories and 1000 attributesin total. Category training and prediction has been completed.


## Project Hierarchy

	└── deep-fashion-classification
	   ├── main.py
	   ├── inference.py
	   ├── model.py
	   ├── requirements.txt
	   ├── README.md
	   └── data
	      ├── category.csv
		  ├── ...
	   └── models
	      ├── category.h5
		  ├── ...
	   └── samples
	      └── pred
		     ├── sample_pred0.png
			 ├── ...
	      ├── blouse_cat3.jpg
		  ├── coat_category39.jpg
		  ├── ...

  
The locations of the files in the project are stated above.

# Usage

### How to predict labels by types

To predict with the trained model:

	python3 main.py --predict --predict-type categories

there are 6 predict types in total:

	'categories'
	'attribute1'	(Textures)
	'attribute2'	(Fabrics)
	'attribute3'	(Shapes)
	'attribute4'	(Parts)
	'attribute5'	(Styles)

### How to train a new model from scratch

Will be added after attributes !!

---
### Model specifications:
- Category:
	- Input Shape: (224, 224, 3) RGB
	- Total params: 11,210,487
	- Trainable params: 7,240,497
	- Non-trainable params: 3,969,990
	- Total layer number: 89


# Training on Colab

Whole model is trained on Google Colab. Takes approximately 16 hours for categories.

https://colab.research.google.com/gist/omerferhatt/93850097ac6def4601a2c728a84c82a6/fashion-attribute-classification.ipynb


# Project Requirements

Library requirements are mentioned in requirements.txt

Required large file links below:

- Training images: https://drive.google.com/file/d/1oMbEqVW16nlxXGLx8TeeQ-evrZMJHcVF/view?usp=sharing
- CSV file for dataset: https://drive.google.com/file/d/19jP57kJ3pI-PkDlXdwQLAV6WvPM2P-z5/view?usp=sharing
- Category Labels: https://drive.google.com/file/d/1go3SOylcSNrX-ZRf6C9Ld0P2FjQ5Ut_K/view?usp=sharing

# Citations

	@inproceedings{liuLQWTcvpr16DeepFashion,
	 author = {Liu, Ziwei and Luo, Ping and Qiu, Shi and Wang, Xiaogang and Tang, Xiaoou},
	 title = {DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations},
	 booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	 month = {June},
	 year = {2016} 
	 }
