# Closed-loop simulation of defined problem
from NMPC import *
from Problem_definition import *
import numpy as np
from casadi import *
import math

NMPC = NMPC()
solver = NMPC.create_solver()
U_pasts, Xd_pasts, Xd_pastse, Xa_pasts, Con_pasts, u_nmpc, time_loop = NMPC.initialization()
number_of_repeats, deltat  = NMPC.number_of_repeats, NMPC.tf/NMPC.nk
simulation_time            = NMPC.simulation_time
time_taken                 = []

for un in range(number_of_repeats):
    loopstart         = time.time()
    arg, u_past, x_hat, Sigma_xhat, xd_previous, t_past, tk, t0i, tfi \
                      = NMPC.initialization_loop()
    Xd_pastse[0,un,:] = x_hat                  
    xu_nominal        = NMPC.mun
    xu_real           = []
    if len(NMPC.mun) != 0:
        xu_real = np.expand_dims(np.random.multivariate_normal(NMPC.mun,NMPC.covun),0).T
    while True:
        # Break when simulation time is reached
        tfi += deltat 
        if round(t0i,2) >= simulation_time:
            break
        
        # Parameter to set initial condition of NMPC algorithm and update discrete time tk
        p, tk    = NMPC.update_inputs(x_hat,tk,u_nmpc)
        arg["p"] = p
        
        # Measure computational time taken
        start    = time.time()
        
        # Solve NMPC problem and extract first control input u_nmpc
        try:
            arg["x0"] = NMPC.load_varsopthyp()
            res       = solver(**arg)
        except:
            arg["x0"] = NMPC.vars_init
            res       = solver(**arg)
        NMPC.save_varsopthyp(np.array(res["x"])[:,0])
        u_nmpc = np.array(NMPC.cfcn(np.array(res["x"])[:,0])) 
        
        # Simulate and measure real system  with this control input
        xd_current, xa_current = NMPC.simulator(xd_previous,u_nmpc,t0i,tfi,xu_real)
        xd_current             = xd_current + np.expand_dims( \
        np.random.multivariate_normal(np.zeros(NMPC.nd),NMPC.Sigma_d),0).T
        for i in range(len(xd_current)):
            if NMPC.state_positive[i]:
                xd_current[i] = np.clip(xd_current[i],0.,inf)
        yd                     = NMPC.hfcn(xd_current) + \
        np.random.multivariate_normal(np.zeros(NMPC.nm),NMPC.Sigma_m)
        x_hat, Sigma_xhat      = NMPC.state_estimator(x_hat,Sigma_xhat,u_nmpc,yd,xu_nominal)
        Xd_pastse[tk+1,un,:]   = x_hat
        xd_previous            = xd_current
        
        # Collect data
        end = time.time()
        t_past, u_past, time_taken = \
        NMPC.collect_data(t_past,u_past,time_taken,start,end,t0i,u_nmpc)
        t0i += deltat
        
    # Generate data for plots
    loopend = time.time()
    Xd_pasts, Xa_Pasts, Con_pasts, U_pasts, time_loop, t_pasts = \
    NMPC.generate_data(Xd_pasts,Xa_pasts,Con_pasts,U_pasts,un,loopend,\
                       time_loop,u_past,xu_real)
    
# Plot results
NMPC.plot_graphs(t_past,t_pasts,Xd_pasts,Xa_Pasts,U_pasts,Xd_pastse,Con_pasts)