'''This script contains the functions needed run simulations in regular grid landscapes.'''


#packages:
import numpy as np
import random as rd
import math
from scipy import signal as signal
from scipy import integrate as integ
import periodic_boundary as pb


#---------------------------------------------------------------------------------------------------------

def landscape_grid(N):
    
    '''
    This function generates a regular grid landscape with N habitats.
    It calculates the habitat distances and saves the values in a landscape matrix "delta" (NxN).
    It returns the habitat coordinates and the landscape matrix.
    '''
    
    # Calculate how many patches are in row for certain total number of habitats
    N_row = int(math.sqrt(N))
    
    # set coordinates
    x_coord = np.repeat(1/N_row * np.arange(N_row), N_row) + 0.5/N_row
    y_coord = np.resize(1/N_row * np.arange(N_row),N_row*N_row) + 0.5/N_row
    
    
    # 2. calculate landscape matrix
    delta = np.zeros((N,N))         # empty matrix, containing only zeros, to be filled with habitat distances
    
    for i in range(N):
        for j in range(N): # loop through habitats
        
            # extract x- and y-coordinates of the habitats
            x_i = x_coord[i]
            x_j = x_coord[j]
            y_i = y_coord[i]
            y_j = y_coord[j]

            # calculate the distances and assign to respective position in matrix
            delta[i,j] = math.sqrt((x_i-x_j)**2 + (y_i-y_j)**2)
    
    # 3. set respective delta matrix and neighbours matrix with periodic boundary conditions
    delta_boundary = pb.delta_matrices(N)
    neighbours = pb.neighbours_matrices(N)
            
    return x_coord, y_coord, delta, delta_boundary, neighbours


#---------------------------------------------------------------------------------------------------------

def linkwise_success(delta, delta_max, neighbours):
    
    '''
    This function calculates the migration success through each link in the landscape. New function for grid landscapes with periodic 
    boudnary conditions - allows self-loops across boundaries.
    Returns the resulting values in a NxN matrix.
    '''
    
    if delta_max == 0: # to prevent runtime error
        success = 1 -  (delta / 0.000001)
    else:
        success = 1 -  (delta / delta_max)
    
    # send values to 0 if...
    for i in range(len(success)):
        for j in range(len(success)):
            for k in range(len(success[i,j])): # for each potential link
                
                # ... negative (= too large distance or delta_max == 0)
                if success[i,j][k] < 0:
                    success[i,j][k] = 0
                    
                # ... diagonals (= no self-loop)
                if i == j: 
                    success[i,j][k] = 0
                    
    return success


#---------------------------------------------------------------------------------------------------------

def fractions_ij(success, neighbours):
    
    '''
    This function calculates the fraction of pollinators migrating towards habitat i among all pollinators emigrating from habitat j. 
    For grid landscapes one patch can several times be neighbour of another patch by using different pathways. Therefore, to get the fraction_ij
    one has to account for all links between i and j.
    '''
    
    # calculate the total migration success for each target-source combination, summing up the single links between them
    success_times_neighbours = success*neighbours
    
    for i in range(len(success)): # select row = target
        for j in range(len(success)): # select col = source
            success_times_neighbours[i,j] = sum(success_times_neighbours[i,j])
            
    denom = sum(success_times_neighbours[:,0]) # all denominators are exactly the same for grid landscapes!
    
    # calculate fraction migrating to i
    if denom == 0:
        fractions = np.zeros((len(success),len(success)))
    
    else:
        fractions = success / denom
        fractions = fractions * neighbours
                
    return fractions


#---------------------------------------------------------------------------------------------------------

def migration_matrix(success, fractions):
    
    '''
    Calculation of migration matrix as combined matrix through elementwise multiplication of migration successes and fractions.
    Through multiplication with the pollinator densities and the dispersal rate, the immigration into the patches is calculated.
    '''
    
    mig_matrix = success * fractions
    for i in range(len(mig_matrix)):
        for j in range(len(mig_matrix)):
            mig_matrix[i,j] = sum(mig_matrix[i,j])
    mig_matrix = mig_matrix.astype('float64')
    
    return mig_matrix


#---------------------------------------------------------------------------------------------------------

def connectance(N, success, neighbours):
    
    '''
    This function is to calculate the connectance of the landscape as realised links / possible links.
    As with periodic boundary conditions the maximum number of links is infinite, I define the number of links
    with delta_max = 1 as the maximum number of links here.
    '''
    
    # calculate maximum number of links
    L_max = 0
    for i in range(N):     # upper right half + diagonal
        for j in range(i, N, 1):
            L_max += sum(neighbours[i,j])
    
    # count number of realised links (upper right triangle)
    L = 0
    for i in range(N):     
        for j in range(i, N, 1):
            for k in range(len(success[i,j])):
                if success[i,j][k] > 0.0:
                    L += neighbours[i,j][k]
                    
    # connectance:
    connect = L / L_max
    
    return connect



#---------------------------------------------------------------------------------------------------------

def overall_migration_success(x, d, mig_matrix, N):
    
    '''
    Calculation of the overall migration success in the landscape as sum of all pollinators immigrating / sum of all 
    pollinators emigrating.
    '''

    # extract the pollinator values from density array
    A = x[-1, ::2]    
        
    # calculate total emigration
    A_emig = d*A
    emig_tot = sum(A_emig)
        
    # calculate total immigration
    A_immig = d* np.dot(mig_matrix, A)
    immig_tot = sum(A_immig)
        
        
    # calculate overall migration success
    if emig_tot > 0:
        f = immig_tot/emig_tot
        
    else:
        f = 0
    
    return f


#---------------------------------------------------------------------------------------------------------

def c_P_values(N, h):
    
    '''
    Function to draw the habitats' plant carrying capacities  K_i from a uniform distribution with range h (symmetrical around 1).  
    Calculation of the respective competition strengthes.
    '''
    
    # open list to store values
    K_values = []
    
    # draw values for N habitats
    for i in range(N):
        K_i = 1 - h/2 + rd.random() * h
    
        K_values.append(K_i)
    
    # calculate competition strenght from carrying capacities
    c_P_vals = np.repeat(1, N)/ K_values
    
    return np.array(c_P_vals)


#---------------------------------------------------------------------------------------------------------

def fixed_points_approx(alpha_r_vals, r_A, r_P_vals, c_A, c_P_vals, gamma_A, gamma_P, d, mig_matrix, N):
    
    '''
    Function to approximatly set the fixed point densities.
    '''
    
    x = []
    
    for i, alpha in enumerate(alpha_r_vals):
        r_P = r_P_vals[i]
        c_P = c_P_vals[i]
        
        # calculate immigration under the assumption that all pollinator populations have the same size
        immig = sum(mig_matrix[i])
    
        A_star = (alpha*gamma_P*r_P + c_P*r_A - d*c_P*(1-immig)) / (c_A* c_P-alpha**2*gamma_A*gamma_P)
        P_star = (alpha*gamma_A*r_A + c_A*r_P - d*alpha*gamma_A*(1-immig)) / (c_A* c_P-alpha**2*gamma_A*gamma_P)

        x.append(A_star)
        x.append(P_star)
        
    return np.array(x)

#----------------------------------------------------------------------------------------

def alpha_max(c_P_vals, c_A, gamma_P, gamma_A):
    
    '''
    This function is to simply calculate the maximum attractiveness alpha_max for one or multiple patches.
    '''
    
    # convert values to arrays:
    c_A = np.asarray(c_A)
    gamma_A = np.asarray(gamma_A)
    gamma_P = np.asarray(gamma_P)
    
    # calculate alpha_max values:
    alpha_max_res = 0.8*np.sqrt((c_A*c_P_vals)/(gamma_A*gamma_P))
        
    
    return alpha_max_res

#--------------------------------------------------------------------------------------

def alpha_optimum(alpha_max_vals, s, c_A, c_P_vals, gamma_A, gamma_P, d, r_A, N, mig_matrix):
    
    '''
    This function is to calculate the optimum alpha value for the given parameter combination.
    It is looped through alpha-values in the possible range and the selection gradient the closest to zero is searched.
    The associated alpha-value is the optimal strategy.
    '''
    
    opt_all = []
    
    for i, alpha_max in enumerate(alpha_max_vals): # loop through the alpha values for the different habitats
        c_P = c_P_vals[i]            # select plant competition coefficient of the respective habitat
        
        # create an array of alpha-values, for which the selection gradient will be calculated (between 0 and alpha_max)
        alpha_vals = np.arange(0.001, alpha_max, 0.001) 
        
        # open empty list to store the values of the selection gradient for the different alpha-values
        f_vals = []
        
        # calculate immigration under the assumption that all pollinator populations have the same size
        immig = sum(mig_matrix[i])
        
        
        # Loop through the alpha-values and calculate the respective selection gradients.
        for alpha in alpha_vals:
            r_P_doa = -(alpha/alpha_max)**s * (1-(alpha/alpha_max)**s)**(1/s) / (alpha * (1-(alpha/alpha_max)**s))
            A_star = (alpha*gamma_P*(1-(alpha/alpha_max)**s)**(1/s) + c_P*r_A - d*c_P*(1-immig)) / (c_A* c_P-alpha**2*gamma_A*gamma_P)
        
            f = r_P_doa + gamma_A * A_star       # selection gradient
            f_vals.append(f)
    
        # Search for the selection gradient the closest to zero, save the index and extract the respective alpha:
        index = signal.argrelmin(abs(np.asarray(f_vals)))
        opt = float(alpha_vals[index])
        
        opt_all.append(opt)
    
    return np.array(opt_all)


#----------------------------------------------------------------------------------------------

def r_P_function(alpha_vals, alpha_max_vals, s):
    
    '''
    This function is to simply calculate the plant growth rates from the alpha-values for multiple patches.
    '''
    
    # open list for the results
    r_P_res = []
    
    for i, alpha in enumerate(alpha_vals): #loop through the alpha values in the different patches
        alpha_max = alpha_max_vals[i]
        
        #calculate and save r_P
        r_P = pow(( 1- pow( alpha/alpha_max ,s)),1/s)
        r_P_res.append(r_P)
        
    return np.array(r_P_res)
              

#---------------------------------------------------------------------------------------------------------------------

def metacommunity(x, t, env_params, c_A, c_P_vals, gamma_A, gamma_P, plant_params, migration_params, N):
    
    '''
    Differential equations defining the dynamical system with plants P and pollinators A growing logistically 
    and interacting on each habitat patch i. Pollinators A can migrate between the patches following emigration
    and immigration terms.
    '''
    
    # extract the parameter values
    r_A0, r_A_change, environmental_change = env_params
    alpha_r_vals, r_P_r_vals = plant_params
    d, mig_matrix = migration_params
    
    # convert to arrays:
    c_A = np.array(c_A)
    gamma_A = np.array(gamma_A)
    gamma_P = np.array(gamma_P)
    
    
    # environmental change (is off for initialisation process):
    if environmental_change:
        r_A = r_A0 + t * r_A_change  
    else:
        r_A = r_A0 + t * 0
        
    # open lists to save results
    x_dot = []
    
    A = x[::2] # even indices are pollinators
    P = x[1::2] # odd indices are plants
        
    # logistic growth
    A_log = r_A *A - c_A*A*A
    P_log = r_P_r_vals *P - c_P_vals*P*P
        
    # mutualistic interaction
    A_int = alpha_r_vals*gamma_P*P*A
    P_int = alpha_r_vals*gamma_A*P*A
        
    # migration
    A_emig = d*A
    A_immig = d* np.dot(mig_matrix, A)
        
    #final change
    A_dot = A_log + A_int - A_emig + A_immig           # logistic growth + interaction benefit + migration
    P_dot = P_log + P_int   # logistic growth + interaction benefit
    
    #save in list
    x_dot.append(A_dot)
    x_dot.append(P_dot)
    
    x_dot = np.array(x_dot).T.flatten() # reshape array that again of form A1, P1, A2, P2,...
    
    return x_dot


#---------------------------------------------------------------------------------------------------------

# function for the evolution
def evolution_function(alpha_r_vals, alpha_m_vals, r_P_r_vals, r_P_m_vals, c_P_vals, r_A, c_A, gamma_P, gamma_A, s, mut_step, d, x):
    
    '''
    This function performs the evolution of the plant attractiveness values alpha.
    With the given values the fitness of the mutant is calculated and if the fitness is positive, the resident strategy is displaced.
    A new mutant stategy is drawn randomly within a defined range close to the resident strategy.
    '''
    
    #open lists to store the new alpha values
    alpha_r_res = []
    alpha_m_res = []
    
    
    for i, alpha_r in enumerate(alpha_r_vals): #loop through the values for the different patches
        
        alpha_m = alpha_m_vals[i]
        r_P_r = r_P_r_vals[i]
        r_P_m = r_P_m_vals[i]
        c_P = c_P_vals[i]
        
        # determine equilibrium densities for resident plants and pollinators
        A_star = x[i*2]   # pollinator
        P_r_star = x[i*2+1]  # resident plant
    
        # Assume a very small density for the mutant population density 
        P_m = 0.000001
    
        # Calculate the mutant's fitness (= per-capita growth rate)
        fitness = r_P_m - c_P*P_m - c_P*P_r_star + alpha_m*gamma_A*A_star
              
        
        # what if P_m survives?
        if fitness > 0.0:
            alpha_r = alpha_m     # new resident population with new alpha
        alpha_m = alpha_r + mut_step*(rd.random()-rd.random())      # new mutants alpha something close to residents alpha
        
        # check if alpha_m is negative, if so send it to zero:
        if alpha_m < 0.0:
            alpha_m = 0
        
        alpha_r_res.append(alpha_r)
        alpha_m_res.append(alpha_m)
        
    return [alpha_r_res, alpha_m_res]

              
    
#---------------------------------------------------------------------------------------------------------

def mutation(alpha_r_vals, mut_step):
    
    
    alpha_m_vals = []
    for alpha_r in alpha_r_vals:
        alpha_m = alpha_r + mut_step * (- 1 + 2 * rd.random())
        
        if alpha_m < 0.0:
            alpha_m = 0
        
        alpha_m_vals.append(alpha_m)
        
    return np.array(alpha_m_vals)


#---------------------------------------------------------------------------------------------------------

def initialisation(c_P_vals, c_A, gamma_P, gamma_A, s, d, r_A0, r_A_change, N, mig_matrix, mut_step):
    
    '''
    This function performs the initialisation of the system after the landscape was set up (i.e. habitat positions, delta matrix, habitat qualities). 
    The optimum attractiveness is approximated, followed by approximation of the fixed point densities. Then the system evolves some time under 
    constant environmental conditions. Returns initial population densities, as well as maximum, resident and mutant attractiveness.
    '''
    
    # calculate alpha_max for all patches
    alpha_max_vals = alpha_max(c_P_vals, c_A, gamma_P, gamma_A)
    
    # calculate alpha_optimum = initial attractiveness (approximate value)
    alpha_r_vals = alpha_optimum(alpha_max_vals, s, c_A, c_P_vals, gamma_A, gamma_P, d, r_A0, N, mig_matrix)
    
    # calculate per-capita growth rate from alpha for fixed point calculation
    r_P_r_vals = r_P_function(alpha_r_vals, alpha_max_vals, s)

    # initial densities (fixed points) -> [A1, P1, A2, P2,...]
    x0 = fixed_points_approx(alpha_r_vals, r_A0, r_P_r_vals, c_A, c_P_vals, gamma_A, gamma_P, d, mig_matrix, N)
    
    # mutant attractiveness values
    alpha_m_vals = mutation(alpha_r_vals, mut_step)
    
    
    #### LET EVOLVE UNDER CONSTANT ENVIRONMENT ####
    t_end = 2000
    t_interval = 20           # length of time interval between mutations
    t_step = 1                # one data point per time step
    
    # empty lists to store results
    t_complete = []
    x_complete = []
    
    for i in range (int(t_end / t_interval)):       # loop through the time intervals
        
        # calculate per-capita growth rates from alpha
        r_P_r_vals = r_P_function(alpha_r_vals, alpha_max_vals, s)
        r_P_m_vals = r_P_function(alpha_m_vals, alpha_max_vals, s)

        # t-values for the current time interval, that is between pertubation i and pertubation i+1:
        t = np.arange(0,t_interval,t_step) + i*t_interval

        # additional parameters to give into the odeint-function
        env_params = [r_A0, r_A_change, False]
        plant_params = [alpha_r_vals, r_P_r_vals]
        migration_params =  [d, mig_matrix]

        # time series
        x = integ.odeint(metacommunity,x0,t, 
                         args=(env_params, c_A, c_P_vals, gamma_A, gamma_P, plant_params, migration_params, N))


        # NO environmental change
        r_A = r_A0 + t * 0
        
        # evolution
        alpha_r_vals, alpha_m_vals = evolution_function(alpha_r_vals, alpha_m_vals, 
                                                                r_P_r_vals, r_P_m_vals, 
                                                                c_P_vals, r_A[-1], 
                                                                c_A, gamma_P, gamma_A, s, 
                                                                mut_step, d, x[-1])
        
        # save densities
        x0 = x[-1]
    
    return x0, alpha_max_vals, alpha_r_vals, alpha_m_vals
