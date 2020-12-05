# Meta-MO
Data and Code for "Meta Learning for Low Resource Molecular Optimization" paper

The 'data' folder contains our bioactive molecular optimization dataset


## Requirements
### Note: Create a new environment for the project, find location of torchtext package, and replace (batch.py, field.py and iterator.py) in data folder of torchtext with corresponding code files in replace_torchtext folder 
python=3.6.10
pytorch=1.1.0
torchtext=0.5.0
rdkit=2017.09.1
networkx=2.4
numpy=1.18.1
...(Install others if required)

## Run the code
1. run prepare_data.py to preprocess the dataset
2. run mo_meta_train.sh to do meta learning model pre-train (mo_all_train.sh -> multi-task pretraining)
3. run mo_meta_dev.sh to do meta validation (mo_all_dev.sh -> multi-task validation)
4. run mo_meta_test.sh to do meta test on test tasks (mo_all_test.sh -> multi-task test)
5. run mo_meta_dev_zs.sh and mo_meta_test_zs.sh to do zero-shot meta learning validation and test (mo_all_dev_zs.sh, mo_all_test_zs.sh -> zeroshot multi-task validation and test)

