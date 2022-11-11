'''
This script contains the landscape matrices and neighbor matrices for the regular grid landscapes with 2x2, 3x3, 4x4 or 5x5 nodes.
The landscape or delta matrix gives the distances between habitats i and j. In the regular grid landscapes with periodic boundary conditions, 
two habitats can be connected via several links. If this is the case, the respective matrix element contains more than one value. 
How many links of each respective length occur between i and j (if delta_max = 1) is given in the neighbors matrices.
'''


#packages:
import numpy as np
import math


#---------------------------------------------------------------------------------------------------------


def delta_matrices(N):
    
    ###################
    if N == 4:
        s0 = np.array([0])
        s = np.array([0.5, 2]) # the 2 is just to make sure the matrix is filled with arrays of the same form, no real neighbour
        d = np.array([0.5*math.sqrt(2)])
        
        matrix = np.array([[s0, s, s, d],
                           [s, s0, d, s],
                           [s, d, s0, s],
                           [d, s, s, s0]],
                         dtype=object)
    
    ###################
    if N == 9:
        s0 = np.array([0])
        s = np.array([1/3, 2/3])
        d = np.array([1/3*math.sqrt(2), 2/3*math.sqrt(2), math.sqrt((1/3)**2+(2/3)**2)])
        
        matrix = np.array([[s0, s, s, s, d, d, s, d, d], 
                           [s, s0, s, d, s, d, d, s, d],
                           [s, s, s0, d, d, s, d, d, s],
                           [s, d, d, s0, s, s, s, d, d],
                           [d, s, d, s, s0, s, d, s, d],
                           [d, d, s, s, s, s0, d, d, s],
                           [s, d, d, s, d, d, s0, s, s],
                           [d, s, d, d, s, d, s, s0, s],
                           [d, d, s, d, d, s, s, s, s0]],
                          dtype=object)
        
    ###################
    if N == 16:
        s0 = np.array([0])
        s1 = np.array([0.25, 0.75])
        s2 = np.array([0.5])
        d1 = np.array([0.25*math.sqrt(2), math.sqrt(0.25**2+0.75**2)])
        d2 = np.array([math.sqrt(0.25**2+0.5**2), math.sqrt(0.5**2+0.75**2)])
        d3 = np.array([0.5*math.sqrt(2)])
        
        matrix = np.array([[s0, s1, s2, s1, s1, d1, d2, d1, s2, d2, d3, d2, s1, d1, d2, d1], 
                           [s1, s0, s1, s2, d1, s1, d1, d2, d2, s2, d2, d3, d1, s1, d1, d2],
                           [s2, s1, s0, s1, d2, d1, s1, d1, d3, d2, s2, d2, d2, d1, s1, d1],
                           [s1, s2, s1, s0, d1, d2, d1, s1, d2, d3, d2, s2, d1, d2, d1, s1],
                           
                           [s1, d1, d2, d1, s0, s1, s2, s1, s1, d1, d2, d1, s2, d2, d3, d2],
                           [d1, s1, d1, d2, s1, s0, s1, s2, d1, s1, d1, d2, d2, s2, d2, d3],
                           [d2, d1, s1, d1, s2, s1, s0, s1, d2, d1, s1, d1, d3, d2, s2, d2],
                           [d1, d2, d1, s1, s1, s2, s1, s0, d1, d2, d1, s1, d2, d3, d2, s2],
                           
                           [s2, d2, d3, d2, s1, d1, d2, d1, s0, s1, s2, s1, s1, d1, d2, d1], 
                           [d2, s2, d2, d3, d1, s1, d1, d2, s1, s0, s1, s2, d1, s1, d1, d2],
                           [d3, d2, s2, d2, d2, d1, s1, d1, s2, s1, s0, s1, d2, d1, s1, d1],
                           [d2, d3, d2, s2, d1, d2, d1, s1, s1, s2, s1, s0, d1, d2, d1, s1],
                           
                           [s1, d1, d2, d1, s2, d2, d3, d2, s1, d1, d2, d1, s0, s1, s2, s1],
                           [d1, s1, d1, d2, d2, s2, d2, d3, d1, s1, d1, d2, s1, s0, s1, s2],
                           [d2, d1, s1, d1, d3, d2, s2, d2, d2, d1, s1, d1, s2, s1, s0, s1],
                           [d1, d2, d1, s1, d2, d3, d2, s2, d1, d2, d1, s1, s1, s2, s1, s0]],
                          dtype=object)
    
    ###################
    if N == 25:
        s0 = np.array([0])
        s1 = np.array([0.2, 0.8])
        s2 = np.array([0.4, 0.6])
        d1 = np.array([0.2*math.sqrt(2), math.sqrt(0.2**2+0.8**2)])
        d2 = np.array([math.sqrt(0.2**2+0.4**2), math.sqrt(0.2**2+0.6**2), math.sqrt(0.8**2+0.4**2)])
        d3 = np.array([0.4*math.sqrt(2), 0.6*math.sqrt(2), math.sqrt(0.4**2+0.6**2)])
        
        matrix = np.array([[s0, s1, s2, s2, s1, s1, d1, d2, d2, d1, s2, d2, d3, d3, d2, s2, d2, d3, d3, d2, s1, d1, d2, d2, d1],
                           [s1, s0, s1, s2, s2, d1, s1, d1, d2, d2, d2, s2, d2, d3, d3, d2, s2, d2, d3, d3, d1, s1, d1, d2, d2],
                           [s2, s1, s0, s1, s2, d2, d1, s1, d1, d2, d3, d2, s2, d2, d3, d3, d2, s2, d2, d3, d2, d1, s1, d1, d2],
                           [s2, s2, s1, s0, s1, d2, d2, d1, s1, d1, d3, d3, d2, s2, d2, d3, d3, d2, s2, d2, d2, d2, d1, s1, d1],
                           [s1, s2, s2, s1, s0, d1, d2, d2, d1, s1, d2, d3, d3, d2, s2, d2, d3, d3, d2, s2, d1, d2, d2, d1, s1],
                           
                           [s1, d1, d2, d2, d1, s0, s1, s2, s2, s1, s1, d1, d2, d2, d1, s2, d2, d3, d3, d2, s2, d2, d3, d3, d2],
                           [d1, s1, d1, d2, d2, s1, s0, s1, s2, s2, d1, s1, d1, d2, d2, d2, s2, d2, d3, d3, d2, s2, d2, d3, d3],
                           [d2, d1, s1, d1, d2, s2, s1, s0, s1, s2, d2, d1, s1, d1, d2, d3, d2, s2, d2, d3, d3, d2, s2, d2, d3],
                           [d2, d2, d1, s1, d1, s2, s2, s1, s0, s1, d2, d2, d1, s1, d1, d3, d3, d2, s2, d2, d3, d3, d2, s2, d2],
                           [d1, d2, d2, d1, s1, s1, s2, s2, s1, s0, d1, d2, d2, d1, s1, d2, d3, d3, d2, s2, d2, d3, d3, d2, s2],
                           
                           [s2, d2, d3, d3, d2, s1, d1, d2, d2, d1, s0, s1, s2, s2, s1, s1, d1, d2, d2, d1, s2, d2, d3, d3, d2],
                           [d2, s2, d2, d3, d3, d1, s1, d1, d2, d2, s1, s0, s1, s2, s2, d1, s1, d1, d2, d2, d2, s2, d2, d3, d3],
                           [d3, d2, s2, d2, d3, d2, d1, s1, d1, d2, s2, s1, s0, s1, s2, d2, d1, s1, d1, d2, d3, d2, s2, d2, d3],
                           [d3, d3, d2, s2, d2, d2, d2, d1, s1, d1, s2, s2, s1, s0, s1, d2, d2, d1, s1, d1, d3, d3, d2, s2, d2],
                           [d2, d3, d3, d2, s2, d1, d2, d2, d1, s1, s1, s2, s2, s1, s0, d1, d2, d2, d1, s1, d2, d3, d3, d2, s2],
                           
                           [s2, d2, d3, d3, d2, s2, d2, d3, d3, d2, s1, d1, d2, d2, d1, s0, s1, s2, s2, s1, s1, d1, d2, d2, d1],
                           [d2, s2, d2, d3, d3, d2, s2, d2, d3, d3, d1, s1, d1, d2, d2, s1, s0, s1, s2, s2, d1, s1, d1, d2, d2],
                           [d3, d2, s2, d2, d3, d3, d2, s2, d2, d3, d2, d1, s1, d1, d2, s2, s1, s0, s1, s2, d2, d1, s1, d1, d2],
                           [d3, d3, d2, s2, d2, d3, d3, d2, s2, d2, d2, d2, d1, s1, d1, s2, s2, s1, s0, s1, d2, d2, d1, s1, d1],
                           [d2, d3, d3, d2, s2, d2, d3, d3, d2, s2, d1, d2, d2, d1, s1, s1, s2, s2, s1, s0, d1, d2, d2, d1, s1],
                           
                           [s1, d1, d2, d2, d1, s2, d2, d3, d3, d2, s2, d2, d3, d3, d2, s1, d1, d2, d2, d1, s0, s1, s2, s2, s1],
                           [d1, s1, d1, d2, d2, d2, s2, d2, d3, d3, d2, s2, d2, d3, d3, d1, s1, d1, d2, d2, s1, s0, s1, s2, s2],
                           [d2, d1, s1, d1, d2, d3, d2, s2, d2, d3, d3, d2, s2, d2, d3, d2, d1, s1, d1, d2, s2, s1, s0, s1, s2],
                           [d2, d2, d1, s1, d1, d3, d3, d2, s2, d2, d3, d3, d2, s2, d2, d2, d2, d1, s1, d1, s2, s2, s1, s0, s1],
                           [d1, d2, d2, d1, s1, d2, d3, d3, d2, s2, d2, d3, d3, d2, s2, d1, d2, d2, d1, s1, s1, s2, s2, s1, s0]],
                          dtype=object)

    
    return matrix

#-------------------------------------------------------------------------------------------------------

def neighbours_matrices(N):
    
    ###################
    if N == 4:
        s0 = np.array([0])
        s = np.array([2, 0])
        d = np.array([4])
        
        matrix = np.array([[s0, s, s, d],
                           [s, s0, d, s],
                           [s, d, s0, s],
                           [d, s, s, s0]],
                         dtype=object)
        
    ###################    
    if N == 9:
        s0 = np.array([0])
        s = np.array([1, 1])
        d = np.array([1, 1, 2])
        
        matrix = np.array([[s0, s, s, s, d, d, s, d, d], 
                           [s, s0, s, d, s, d, d, s, d],
                           [s, s, s0, d, d, s, d, d, s],
                           [s, d, d, s0, s, s, s, d, d],
                           [d, s, d, s, s0, s, d, s, d],
                           [d, d, s, s, s, s0, d, d, s],
                           [s, d, d, s, d, d, s0, s, s],
                           [d, s, d, d, s, d, s, s0, s],
                           [d, d, s, d, d, s, s, s, s0]],
                          dtype=object)
    
    ###################
    if N == 16:
        s0 = np.array([0])
        s1 = np.array([1, 1])
        s2 = np.array([2])
        d1 = np.array([1, 2])
        d2 = np.array([2, 2])
        d3 = np.array([4])
        
        matrix = np.array([[s0, s1, s2, s1, s1, d1, d2, d1, s2, d2, d3, d2, s1, d1, d2, d1], 
                           [s1, s0, s1, s2, d1, s1, d1, d2, d2, s2, d2, d3, d1, s1, d1, d2],
                           [s2, s1, s0, s1, d2, d1, s1, d1, d3, d2, s2, d2, d2, d1, s1, d1],
                           [s1, s2, s1, s0, d1, d2, d1, s1, d2, d3, d2, s2, d1, d2, d1, s1],
                           
                           [s1, d1, d2, d1, s0, s1, s2, s1, s1, d1, d2, d1, s2, d2, d3, d2],
                           [d1, s1, d1, d2, s1, s0, s1, s2, d1, s1, d1, d2, d2, s2, d2, d3],
                           [d2, d1, s1, d1, s2, s1, s0, s1, d2, d1, s1, d1, d3, d2, s2, d2],
                           [d1, d2, d1, s1, s1, s2, s1, s0, d1, d2, d1, s1, d2, d3, d2, s2],
                           
                           [s2, d2, d3, d2, s1, d1, d2, d1, s0, s1, s2, s1, s1, d1, d2, d1], 
                           [d2, s2, d2, d3, d1, s1, d1, d2, s1, s0, s1, s2, d1, s1, d1, d2],
                           [d3, d2, s2, d2, d2, d1, s1, d1, s2, s1, s0, s1, d2, d1, s1, d1],
                           [d2, d3, d2, s2, d1, d2, d1, s1, s1, s2, s1, s0, d1, d2, d1, s1],
                           
                           [s1, d1, d2, d1, s2, d2, d3, d2, s1, d1, d2, d1, s0, s1, s2, s1],
                           [d1, s1, d1, d2, d2, s2, d2, d3, d1, s1, d1, d2, s1, s0, s1, s2],
                           [d2, d1, s1, d1, d3, d2, s2, d2, d2, d1, s1, d1, s2, s1, s0, s1],
                           [d1, d2, d1, s1, d2, d3, d2, s2, d1, d2, d1, s1, s1, s2, s1, s0]],
                          dtype=object)

    ###################    
    if N == 25:
        s0 = np.array([0])
        s1 = np.array([1, 1])
        s2 = np.array([1, 1])
        d1 = np.array([1, 2])
        d2 = np.array([1, 1, 1])
        d3 = np.array([1, 1, 2])
        
        matrix = np.array([[s0, s1, s2, s2, s1, s1, d1, d2, d2, d1, s2, d2, d3, d3, d2, s2, d2, d3, d3, d2, s1, d1, d2, d2, d1],
                           [s1, s0, s1, s2, s2, d1, s1, d1, d2, d2, d2, s2, d2, d3, d3, d2, s2, d2, d3, d3, d1, s1, d1, d2, d2],
                           [s2, s1, s0, s1, s2, d2, d1, s1, d1, d2, d3, d2, s2, d2, d3, d3, d2, s2, d2, d3, d2, d1, s1, d1, d2],
                           [s2, s2, s1, s0, s1, d2, d2, d1, s1, d1, d3, d3, d2, s2, d2, d3, d3, d2, s2, d2, d2, d2, d1, s1, d1],
                           [s1, s2, s2, s1, s0, d1, d2, d2, d1, s1, d2, d3, d3, d2, s2, d2, d3, d3, d2, s2, d1, d2, d2, d1, s1],
                           
                           [s1, d1, d2, d2, d1, s0, s1, s2, s2, s1, s1, d1, d2, d2, d1, s2, d2, d3, d3, d2, s2, d2, d3, d3, d2],
                           [d1, s1, d1, d2, d2, s1, s0, s1, s2, s2, d1, s1, d1, d2, d2, d2, s2, d2, d3, d3, d2, s2, d2, d3, d3],
                           [d2, d1, s1, d1, d2, s2, s1, s0, s1, s2, d2, d1, s1, d1, d2, d3, d2, s2, d2, d3, d3, d2, s2, d2, d3],
                           [d2, d2, d1, s1, d1, s2, s2, s1, s0, s1, d2, d2, d1, s1, d1, d3, d3, d2, s2, d2, d3, d3, d2, s2, d2],
                           [d1, d2, d2, d1, s1, s1, s2, s2, s1, s0, d1, d2, d2, d1, s1, d2, d3, d3, d2, s2, d2, d3, d3, d2, s2],
                           
                           [s2, d2, d3, d3, d2, s1, d1, d2, d2, d1, s0, s1, s2, s2, s1, s1, d1, d2, d2, d1, s2, d2, d3, d3, d2],
                           [d2, s2, d2, d3, d3, d1, s1, d1, d2, d2, s1, s0, s1, s2, s2, d1, s1, d1, d2, d2, d2, s2, d2, d3, d3],
                           [d3, d2, s2, d2, d3, d2, d1, s1, d1, d2, s2, s1, s0, s1, s2, d2, d1, s1, d1, d2, d3, d2, s2, d2, d3],
                           [d3, d3, d2, s2, d2, d2, d2, d1, s1, d1, s2, s2, s1, s0, s1, d2, d2, d1, s1, d1, d3, d3, d2, s2, d2],
                           [d2, d3, d3, d2, s2, d1, d2, d2, d1, s1, s1, s2, s2, s1, s0, d1, d2, d2, d1, s1, d2, d3, d3, d2, s2],
                           
                           [s2, d2, d3, d3, d2, s2, d2, d3, d3, d2, s1, d1, d2, d2, d1, s0, s1, s2, s2, s1, s1, d1, d2, d2, d1],
                           [d2, s2, d2, d3, d3, d2, s2, d2, d3, d3, d1, s1, d1, d2, d2, s1, s0, s1, s2, s2, d1, s1, d1, d2, d2],
                           [d3, d2, s2, d2, d3, d3, d2, s2, d2, d3, d2, d1, s1, d1, d2, s2, s1, s0, s1, s2, d2, d1, s1, d1, d2],
                           [d3, d3, d2, s2, d2, d3, d3, d2, s2, d2, d2, d2, d1, s1, d1, s2, s2, s1, s0, s1, d2, d2, d1, s1, d1],
                           [d2, d3, d3, d2, s2, d2, d3, d3, d2, s2, d1, d2, d2, d1, s1, s1, s2, s2, s1, s0, d1, d2, d2, d1, s1],
                           
                           [s1, d1, d2, d2, d1, s2, d2, d3, d3, d2, s2, d2, d3, d3, d2, s1, d1, d2, d2, d1, s0, s1, s2, s2, s1],
                           [d1, s1, d1, d2, d2, d2, s2, d2, d3, d3, d2, s2, d2, d3, d3, d1, s1, d1, d2, d2, s1, s0, s1, s2, s2],
                           [d2, d1, s1, d1, d2, d3, d2, s2, d2, d3, d3, d2, s2, d2, d3, d2, d1, s1, d1, d2, s2, s1, s0, s1, s2],
                           [d2, d2, d1, s1, d1, d3, d3, d2, s2, d2, d3, d3, d2, s2, d2, d2, d2, d1, s1, d1, s2, s2, s1, s0, s1],
                           [d1, d2, d2, d1, s1, d2, d3, d3, d2, s2, d2, d3, d3, d2, s2, d1, d2, d2, d1, s1, s1, s2, s2, s1, s0]],
                          dtype=object)
        
    return matrix


