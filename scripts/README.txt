HIGGS BOSON PROJECT 1

##################################
AUTHORS: Costanza Volpini, Pedro Abranches, Youssef Kitane
##################################

DESCRIPTION:
The aim of this project is to be able to build a robust machine learning model that is able to distinguish between background noise and the possible signal from Higgs boson.
Machine learning methods were applied to simulate the discovering of the Higgs particle.

STRUCTURE:
- datas/ : folder containing all the dataset provided for this project (test.csv and train.csv)
- documents/ : folder containing the report, project description, project guideline files
- scripts/ : folder containing all the python scripts used in the project

PYTHON SCRIPTS:
- cross_validation.py : functions to apply cross validation technique given a method and a dataset
- execute_code.py : functions to execute one method or more with/without cross validation and to generate a submission
- helpers_functions.py : contains helper functions to calculate loss, mean, median, accuracy, gradient and generate minibatch iterator.
- implementations.py : machine learning baseline functions
- spli_jet_num.py : functions to split the dataset in 3 parts by jet_num feature and to drop columns containing constant values
- pre_processing.py : functions to clean dataset and make feature augmentation.
- proj1_helpers.py : helper functions specifically for this project
- run.py : simulate the code to get our best result

TO RUN THE CODE:
1. Install numpy (pip3 install numpy for OSX)
2. go into folder scripts
3. python3 run.py