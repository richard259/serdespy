# serdespy
A python library for system-level SerDes modelling and simulation

## Table of Contents
* [General Info](#general-information)
* [Features](#features)
* [Setup](#setup)
* [Examples](#examples)
* [Project Status](#project-status)

## General Information

Richard Barrie's undergraduate thesis project. University of Toronto, Engineering Science. ESC499 2021/2022. 
Supervisors: Tony Chan Causone, Ming Yang
Authors: Richard Barrie, Katherine Liang

## Features
Includes functions and classes for time-domain model of serdes system
- Channel Modelling
- TX FIR Filter
- TX Jitter
- Continuous-Time Linear Equalizer (CTLE)
- Feed-Forward Equalizer (FFE)
- Decision Feedback Equalizer (DFE)
- Maximum-Likelihood Sequence Estimation (MLSE)
- PRBS/PRQS generation
- Eye Diagram Plotter
- Bit Error Rate Checker
- Forward Error Correction with Reed-Solomon Codes

## Setup
Python 3.7+ required
`pip install serdespy`

## Examples
The 'examples' directory contains some scripts and documentation on how to use serdespy. It contains 2 subdirectories:

- /problem_set
    - contains a problem set on wireline links "problem_set.pdf", and scripts that complete the problems using the serdespy package
    - files should be run in numerical order

- /example_channel
    - this directory contains a series of scripts that perform modelling and simulation on a model of a 100G PAM-4 copper channel from 4-port s-paramaters
    - instructions for downloading the s-paramater files and running the scripts are in the header of "1_channel.py"
    - files should be run in numerical order
