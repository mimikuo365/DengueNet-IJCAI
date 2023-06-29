# DengueNet

This is the repo for the IJCAI workshop paper entitled "DengueNet: Dengue Prediction using Spatiotemporal Satellite Imagery for Resource-Limited Countries".

## File Structure

- model: Contains all the model (a.k.a. DengueNet) implementation
- py_script: Contains the main python scripts to run
  - Demo.py: Plot images and results for demo
  - Evaluate.py: Evaluate stored models' performance
  - Train.py: Read in the dataset and train models
- util: Contains the helper functions

## Dataset Structure

The dataset contains images from five Colombian municipalities. Each municipality has 156 records of data.
- Medellin
  - feature
    - 201601.pickle
    - 201602.pickle
    ...
    - 201852.pickle
  - image
    - 201601.tiff
    - 201602.tiff
    ...
    - 201852.tiff
- Ibague
- Cali
- Villavicencio
- Cucuta
