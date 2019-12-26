Your task is to code and train an image segmentation model that
can segment the blood vessels of a retina image from the DRIVE dataset.
This is very important clinically as they have been shown to be a strong
biomarker of diseases such as diabetic retinopathy.

I have already written up most of the code, but it is your task to complete
it and then train the model of your choice.

The files you are responsible for completing are the following:
  - train.py
  - eval.py
  - data/dataset.py
  - models/model.py

And to run the files, you should edit the following and change hyperparameters:
  - train.sh : to train the model
  - test.sh : to test the model

Note, we use the test set as our validation set. Furthermore, you are to complete a
function in eval.py that stores the predicted segmentations as a .png in which you can view
and can show us when you're done this project.

We are giving you the flexibility to do some research and implement which ever model you feel
is appropriate for this task. It is also an opportunity for you to get creative and come up
with something new or cool if you wish.

Data augmentation is very important especially considering our dataset has only 20 images.
Please implement and use as much data augmentation as possible.

If you have any questions, need any clarifications, or there are mistakes in the code, please
contact me at j294sun@uwaterloo.

Best of luck!
