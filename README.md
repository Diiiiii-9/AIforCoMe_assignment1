# AI for Computational Mechanics - Assignment 1

**Technical University of Munich (TUM)** **Task:** Development of a Neural Network for Microstructure-Property Predictions

## Project Overview
This repository contains the codebase and report for Assignment 1. The objective is to develop a lightweight, efficient Convolutional Neural Network (CNN) to predict the effective Young's modulus ($E_{eff}$) of composite materials directly from $65 \times 65$ binary microstructure images.

The primary goal is to achieve a relative prediction error of less than 30% on an unseen test set while maximizing model efficiency (minimizing training data usage and model parameters).

## Team Members
* Di Liu
* Huixin Zhang

## Repository Structure
* `data/`: Contains the `.npy` dataset (Ignored by Git).
* `models/`: Stores the trained `.pth` model weights (Ignored by Git).
* `src/`: Core Python scripts for data loading (`dataset.py`), model architecture (`model.py`), training (`train.py`), and evaluation (`evaluate.py`).
* `notebooks/`: Exploratory Data Analysis (EDA) and prototyping.
* `report/`: LaTeX files for the final 2-page submission.