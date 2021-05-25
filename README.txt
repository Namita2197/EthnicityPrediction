Namita Raghuvanshi
A02310449

Requirements- 
python version - 3.6

numpy==1.16.4
matplotlib==3.0.2
scikit-learn==0.23.1
tensorflow==2.3.1
pickle

(Folder Hierarchy of what I have submitted):
README.txt
FinalReport.pdf
Source Code(this is a pycharm project)/
	ethnicityClassification.py(source code for ethnicity classification)
	genderClassification.py(source code for gender classification)
	genderPickleFiles/
		X_train.pck
		X_test.pck
		X_valid.pck
		y_train.pck
		y_test.pck
		y_valid.pck
	ethynicityPickleFiles/
		X_train.pck
		X_test.pck
		X_valid.pck
		y_train.pck
		y_test.pck
		y_valid.pck
	BestGenderClassification/
		(contains the best saved model) 
	BestEthnicityClassification/
		(contains the best saved model)

To run the code-
I am submitting a pycharm project(Source code).
If you run this without changing anything it should generate the final validation accuracies of the saved model.
To start training, uncomment the commented code.

All the dependent folders are inside the project folder so that you dont need to worry about the paths to load the data or the saved models.

X_train and y_train ------> training purpose
X_valid and y_valid ------> validation after every epoch
X_test  and y_test  ------> Final validation(reported accuracies are predicted on this data)