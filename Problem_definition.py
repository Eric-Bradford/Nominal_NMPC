# NMPC model and problem setup
import numpy as np
from casadi import *
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints as MSSP

def specifications():
    
    # Initial conditions
    CA0        = 2.    
    CB0        = 0.    
    CC0        = 0.    
    Vol0       = 0.5   
    CBin0      = 10.
    x0         = np.array([CA0,CB0,CC0,Vol0,CBin0]) # initial state of plant
    
    # Initial state estimates
    CAhat0     = 1.5    
    CBhat0     = 0.    
    CChat0     = 0.    
    Volhat0    = 0.2   
    CBinhat0   = 2.
    xhat0      = np.array([CAhat0,CBhat0,CChat0,Volhat0,CBinhat0])    # initial state estimate
    Sigmaxhat0 = 0.01*np.diag([CAhat0,0.1,0.1,Volhat0,CBinhat0*100.]) # initial state covariance estimate
    
    # NMPC algorithm
    tf = 4.                  # time horizon
    nk = 30                  # number of control intervals 
    shrinking_horizon = True # shrinking horizon or receding horizon
    
    # Discredization using direct collocation 
    deg  = 4       # Degree of interpolating polynomialc
    cp   = "radau" # Type of collocation points
    nicp = 1       # Number of (intermediate) collocation points per control interval
    
    # Simulation
    simulation_time   = 4. # simulation time
    number_of_repeats = 1  # number of Monte Carlo simulations  
    
    # NLP solver
    opts                                = {}
    opts["expand"]                      = True
    opts["ipopt.max_iter"]              = 10000
    opts["ipopt.tol"]                   = 1e-10
    opts['ipopt.linear_solver']         = 'mumps' # recommended to use ma27

    return x0, tf, nk, shrinking_horizon, deg, cp, nicp, simulation_time,\
opts, number_of_repeats, xhat0, Sigmaxhat0

def DAE_system():
    
    # Define vectors with strings of state, algebraic and input variables
    states         = ['CA','CB','CC','Vol','CBin'] 
    algebraics     = []
    inputs         = ['F']
    state_positive = [True,True,True,True,True] # State is >= 0 for state estimator
    
    # Define model parameter names and values
    modpar      = ['k']
    modparval   = [2.]
    
    # Define uncertain parameter names and values
    unpar       = []
    nun         = len(unpar)
    mun         = np.array([])           # Nominal value of uncertain parameters
    covun       = np.empty(shape=(0, 0)) # Covariance of uncertain parameters
    
    xu          = SX.sym('xu',nun)  
    for i in range(nun):
        globals()[unpar[i]] = xu[i]
    
    nd          = len(states)
    xd          = SX.sym('xd',nd)  
    for i in range(nd):
        globals()[states[i]] = xd[i]
        
    na          = len(algebraics)
    xa          = SX.sym("xa",na)       
    for i in range(na):
        globals()[algebraics[i]] = xa[i]
        
    nu          = len(inputs)
    u           = SX.sym("u",nu)
    for i in range(nu):
        globals()[inputs[i]] = u[i]
        
    nmp         = len(modpar)
    for i in range(nmp):
        globals()[modpar[i]] = SX(modparval[i])
    
    # Declare ODE equations (use notation as defined in the strings)
    dCA   = -k*CA*CB - F*CA/Vol
    dCB   = -k*CA*CB + F*(CBin-CB)/Vol
    dCC   =  k*CA*CB - F*CC/Vol
    dVol  =  F
    dCBin = 0.
    ODEeq = [dCA,dCB,dCC,dVol,dCBin]  
    
    # Declare Algebraic equations
    Aeq   =  []
    
    # Define objective 
    Obj_M       = Function('mayer', [xd,xa,u],[-CC*Vol]) # Mayer term
    Obj_L       = Function('lagrange', [xd,xa,u],[0.])   # Lagrange term
    R           = 0.1*SX.eye(nu)                         # Weighting of control penality                   
    
    # Define control bounds
    u_min = np.array([0.])
    u_max = np.array([0.4])
    
    # Define constraint functions g(x) <= 0 (True=path, False=terminal)
    gequation = vertcat(Vol-0.75,CB-1.0)
    ng        = SX.size(gequation)[0]
    gfcn      = Function('gfcn',[xd,xa,u],[gequation])
    path      = [False,True]

    # Matrix for weighting the soft-constraints
    G = 10**6*diag(SX.ones(ng))
    
    # Additive disturbance and measurement noise 
    nm      = 3 # number of measurements
    Sigma_d = 1e-3*np.diag(np.ones(nd))
    Sigma_m = 5e-3*np.diag(np.ones(nm))

    # Measurement model
    def hfcn(x):
        y = np.hstack((x[0],x[1],x[3]))
        return y

    # Disturbance and measurement noise for state estimator
    Sigma_Q = 1e-2*np.diag(np.array([1.]*4 + [100.]))
    Sigma_R = Sigma_m

    return xd, xa, u, ODEeq, Aeq, Obj_M, Obj_L, R, ng, gfcn, G, u_min, state_positive, \
u_max, states, algebraics, inputs, Sigma_d, Sigma_m, nm, xu, mun, covun, path, hfcn, Sigma_Q, Sigma_R    
    
def state_estimator(xhat,Sigmaxhat,uNMPC,yd,xu_nominal):
    
    xd, xa, u, ODEeq, Aeq, Obj_M, Obj_L, R, ng, gfcn, G, u_min, state_positive, \
u_max, states, algebraics, inputs, Sigma_d, Sigma_m, nm, xu, mun, covun, path, hfcn, Sigma_Q, Sigma_R\
                = DAE_system()
    x0, tf, nk, shrinking_horizon, deg, cp, nicp, simulation_time,\
opts, number_of_repeats, xhat0, Sigmahat0 = specifications()            
    nd, na, nu  = SX.size(xd)[0], SX.size(xa)[0], SX.size(u)[0]
    deltat      = tf/nk
    
    def fx(xcon,dt):
        
        for i in range(nd):
            if state_positive[i]:
                xcon[i] = np.clip(xcon[i],0.,inf)
        
        ODE = []
        for i in range(nd):
            ODE = vertcat(ODE,substitute(ODEeq[i],vertcat(u,xu),vertcat(SX(uNMPC),SX(xu_nominal)))) 
        
        A = []
        for i in range(na):
            A   = vertcat(A,substitute(Aeq[i],vertcat(u,xu),vertcat(SX(uNMPC),SX(xu_nominal))))
        
        dae = {'x':xd, 'z':xa, 'ode':ODE, 'alg':A}        
        I = integrator('I', 'idas', dae, {'t0':0., 'tf':deltat, 'abstol':1e-10, \
        'reltol':1e-10})
        res = I(x0=xcon)
        xd_current = np.array(res['xf']).flatten()
        
        return xd_current
    
    points = MSSP(nd,alpha=1e-1,beta=1.,kappa=0.)
    ukf    = UKF(dim_x=nd,dim_z=nm,dt=1.,fx=fx,hx=hfcn,points=points)
    ukf.x  = xhat
    ukf.P  = Sigmaxhat
    ukf.Q  = Sigma_Q
    ukf.R  = Sigma_R
    ukf.predict()
    ukf.update(yd)
    newxhat      = ukf.x
    newSigmaxhat = ukf.P
    for i in range(nd):
        if state_positive[i]:
            newxhat[i] = np.clip(newxhat[i],0.,inf)
    
    return newxhat, newSigmaxhat 