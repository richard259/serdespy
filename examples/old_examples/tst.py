# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 20:19:46 2021

@author: Richard Barrie
"""

from serdespy import *



simple_eye(nrz_input(10,prbs13(1),[-0.5,0.5]), 10*3, 100, 1e-12, "eye diagram (ideal)")


