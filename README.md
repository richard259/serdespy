# serdespy
> A python library for system-level SerDes modelling and simulation

## Table of Contents
* [General Info](#general-information)
* [Features](#features)
* [Setup](#setup)
* [Description](#description)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Contact](#contact)
<!-- * [License](#license) -->


## General Information

Richard Barrie's undergraduate thesis project. University of Toronto, Engineering Science. ESC499 2021/2022. 
Supervisors: Tony Chan Causone, Ming Yang
Authors: Richard Barrie, Katherine Liang


## Features
Includes functions and classes for time-domain model of serdes system
- Channel Modelling
- TX FIR
- TX Jitter
- Continuous-Time Linear Equalizer
- Feed-Forward Equalizer
- Decision Feedback Equalizer
- PRBS/PRQS generation
- Eye Diagram Plotter
- Bit Error Rate Checker
- Forward Error Correction with Reed-Solomon Codes

## Setup
Python 3.7+ required
`pip install serdespy`

## Description

/serdespy
- python module containing functions and classes for SerDes Modelling

/examples 
- example of serdes model and simulation using serdespy package

/touchstone
- 4-port touchstone files from https://www.ieee802.org/3/ck/public/tools/index.html

/test_signals
- pre-generated PRBS signals saved in numpy arrays

## Project Status
Project is in progress
Feel free raise an issue or make a pull request with an addition!

## Future Work
Planned additions to the package include:
- clock and data recovery model
- multiprocessing for long BER test simulations (with FEC)
- statistical modelling


## Contact
richard.barrie@mail.utoronto.ca


