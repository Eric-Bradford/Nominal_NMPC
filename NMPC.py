from pylab import *
import numpy as np
import math
from Problem_definition import *
from casadi import *
from scipy.io import savemat
import pickle

class NMPC:
    def __init__(self):
        # Variable definitions
        self.xd, self.xa, self.u, self.ODEeq, self.Aeq, self.Obj_M, \
        self.Obj_L, self.R, self.ng, self.gfcn, self.G, self.u_min, self.state_positive, \
        self.u_max, self.states, self.algebraics, self.inputs, \
        self.Sigma_d, self.Sigma_m, self.nm, self.xu, self.mun, self.covun,\
        self.path, self.hfcn, self.Sigma_Q, self.Sigma_R = DAE_system()
        self.x0, self.tf, self.nk, self.shrinking_horizon, self.deg, \
        self.cp, self.nicp, self.simulation_time, self.opts, \
        self.number_of_repeats, self.xhat0, self.Sigmahat0 = specifications()
        self.h = self.tf/self.nk/self.nicp 
        self.nd, self.na = SX.size(self.xd)[0], SX.size(self.xa)[0] 
        self.nu   = SX.size(self.u)[0]
        self.state_estimator = state_estimator
        
        # Internal function calls
        self.C, self.D            = self.collocation_points()
        self.ffcn                 = self.model_fcn()
        self.NV, self.V, self.vars_lb, self.vars_ub, self.vars_init, self.XD, \
        self.XA, self.U, self.con = self.NLP_specification()   
        self.vars_init, self.vars_lb, self.vars_ub, self.g, self.lbg, self.ubg, \
        self.lambdav, self.XD, self.XA, self.U, self.cfcn, self.lambdac \
                                  = self.set_constraints()
        self.Obj                  = self.set_objective()
        self.solver               = self.create_solver()
    
    def collocation_points(self):
        deg, cp, nk, h = self.deg, self.cp, self.nk, self.h
        C = np.zeros((deg+1,deg+1)) # Coefficients of the collocation equation
        D = np.zeros(deg+1)         # Coefficients of the continuity equation
        
        # All collocation time points
        tau = SX.sym("tau") # Collocation point
        tau_root = [0] + collocation_points(deg,cp)
        T = np.zeros((nk,deg+1))
        for i in range(nk):
            for j in range(deg+1):
                T[i][j] = h*(i + tau_root[j])
        # For all collocation points: eq 10.4 or 10.17 in Biegler's book
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        for j in range(deg+1):
            L = 1
            for j2 in range(deg+1):
                if j2 != j:
                    L *= (tau-tau_root[j2])/(tau_root[j]-tau_root[j2])
            lfcn = Function('lfcn', [tau],[L])
        
            # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
            D[j] = lfcn(1.0)
            # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
            tfcn = Function('tfcn', [tau],[tangent(L,tau)])
            for j2 in range(deg+1):
                C[j][j2] = tfcn(tau_root[j2]) 
            
        return C, D
    
    def model_fcn(self):
        xd, xa, u, xu, ODEeq, Aeq = self.xd, self.xa, self.u, self.xu, self.ODEeq, self.Aeq
        t     =   SX.sym("t")               
        p_s   =   SX.sym("p_s")             
        xddot =   SX.sym("xddot",self.nd)  
        
        res   = []
        for i in range(self.nd):
            res = vertcat(res,ODEeq[i]*p_s - xddot[i]) 
        
        for i in range(self.na):
            res = vertcat(res,Aeq[i]*p_s)
        
        ffcn = Function('ffcn', [t,xddot,xd,xa,xu,u,p_s],[res])
        
        return ffcn

    def NLP_specification(self):
        xd, xa, u, nk, deg, nicp = self.xd, self.xa, self.u, self.nk, self.deg, self.nicp
        nd, na, nu, nx           = self.nd, self.na, self.nu, self.nd+self.na 
        ng, gfcn                 = self.ng, self.gfcn
        nicp, deg                = self.nicp, self.deg
        
        # Total number of variables
        NXD = nicp*nk*(deg+1)*nd  # Collocated differential states
        NXA = nicp*nk*deg*na      # Collocated algebraic states
        NU  = nk*nu               # Parametrized controls
        NV  = NXD+NXA+NU
        
        # NLP variable vector
        V   =   MX.sym("V",NV+nk*ng)
        con =   MX.sym("con",nd+nk+nu)
        
        # All variables with bounds and initial guess
        vars_lb   = np.zeros(NV+nk*ng)
        vars_ub   = np.zeros(NV+nk*ng)
        vars_init = np.zeros(NV+nk*ng)
        
        # differential states, algebraic states and control matrix definition after
        # discredization
        XD      = np.resize(np.array([],dtype=MX),(nk,nicp,deg+1)) # NB: same name as above
        XA      = np.resize(np.array([],dtype=MX),(nk,nicp,deg)) # NB: same name as above
        U       = np.resize(np.array([],dtype=MX),nk)
        
        return NV, V, vars_lb, vars_ub, vars_init, XD, XA, U, con

    def set_constraints(self):
        
        nk, nicp, deg, C, h   = self.nk, self.nicp, self.deg, self.C, self.h
        ffcn, D               = self.ffcn, self.D
        nd, na, nu, nx        = self.nd, self.na, self.nu, self.nd+self.na
        u_min,u_max           = self.u_min, self.u_max
        ng, gfcn, mun         = self.ng, self.gfcn, self.mun
        con, V, NV, path      = self.con, self.V, self.NV, self.path
        vars_lb, vars_ub      = self.vars_lb, self.vars_ub
        XD, XA, U, vars_init  = self.XD, self.XA, self.U, self.vars_init
        
        lambdac       = [MX.zeros(ng)]*nk
        x_current     = con[:nd]
        p_s           = con[nd:nd+nk]
        lambdav       = V[-nk*ng:]
        
        xD_init          = np.array((nk*nicp*(deg+1))*[[1.]*nd])
        xA_init          = np.array((nk*nicp*(deg+1))*[[1.]*na])
        u_init           = np.array((nk*nicp*(deg+1))*[[1.]*nu])
        vars_lb[-nk*ng:] = np.zeros(nk*ng)
        vars_ub[-nk*ng:] = np.ones(nk*ng)*inf
        
        offset  = 0
        
        xD_min, xD_max  = np.array([-inf]*nx), np.array([inf]*nx)
        xDf_min,xDf_max = np.array([-inf]*nx), np.array([inf]*nx)
        xA_min, xA_max  = np.array([-inf]*na), np.array([inf]*na)
        
        # Get collocated states and parametrized control
        for k in range(nk):  
            # Collocated states
            for i in range(nicp):
                #
                for j in range(deg+1):
                              
                    # Get the expression for the state vector
                    XD[k][i][j] = V[offset:offset+nd]
                    if j !=0:
                        XA[k][i][j-1] = V[offset+nd:offset+nd+na]
                    # Add the initial condition
                    index = (deg+1)*(nicp*k+i) + j
                    if k==0 and j==0 and i==0:
                        vars_init[offset:offset+nd] = xD_init[index,:]
                        
                        vars_lb[offset:offset+nd] = xD_min
                        vars_ub[offset:offset+nd] = xD_max                    
                        offset += nd
                    else:
                        if j!=0:
                            vars_init[offset:offset+nx] = np.append(xD_init[index,:],xA_init[index,:]) 
                            
                            vars_lb[offset:offset+nx] = np.append(xD_min,xA_min)
                            vars_ub[offset:offset+nx] = np.append(xD_max,xA_max)
                            offset += nx
                        else:
                            vars_init[offset:offset+nd] = xD_init[index,:]
                            
                            vars_lb[offset:offset+nd] = xD_min
                            vars_ub[offset:offset+nd] = xD_max
                            offset += nd
            
            # Parametrized controls
            U[k]                        = V[offset:offset+nu]
            vars_lb[offset:offset+nu]   = u_min
            vars_ub[offset:offset+nu]   = u_max
            vars_init[offset:offset+nu] = u_init[index,:]
            offset                     += nu
        
        assert(offset==NV)
        
        # Constraint function for the NLP
        g = []
        lbg = []
        ubg = []
        
        # Initial value constraint
        g   +=  [XD[0][0][0] - x_current]
        lbg.append(np.zeros(nd)) 
        ubg.append(np.zeros(nd)) 

        # For all finite elements
        for k in range(nk):
            for i in range(nicp):
                # For all collocation points
                for j in range(1,deg+1):                
                    # Get an expression for the state derivative at the collocation point
                    xp_jk = 0
                    for j2 in range (deg+1):
                        xp_jk += C[j2][j]*XD[k][i][j2]       # get the time derivative of the differential states (eq 10.19b)
                    
                    # Add collocation equations to the NLP
                    fk = ffcn(0.,xp_jk/h,XD[k][i][j],XA[k][i][j-1],MX(mun),U[k],p_s[k])
                    g += [fk[:nd]]           # impose system dynamics (for the differential states (eq 10.19b))
                    lbg.append(np.zeros(nd)) # equality constraints
                    ubg.append(np.zeros(nd)) # equality constraints
                    g += [fk[nd:]]                               # impose system dynamics (for the algebraic states (eq 10.19b))
                    lbg.append(np.zeros(na)) # equality constraints
                    ubg.append(np.zeros(na)) # equality constraints
                                                                           
                np.resize(np.array([],dtype=SX),(nk,nicp,deg))
                # Get an expression for the state at the end of the finite element
                if k > 0:
                    xf_k = 0
                    for j in range(deg+1):
                        xf_k += D[j]*XD[k-1][i][j]
                    
                    # Add continuity equation to NLP
                    if i==nicp-1:
                        g += [XD[k][0][0] - xf_k]
                    else:
                        g += [XD[k-1][i+1][0] - xf_k]
                
                    lbg.append(np.zeros(nd))
                    ubg.append(np.zeros(nd))
                        
        cfcn = Function('cfcn',[V],[U[0]])
        
        offset2 = 0
        for gg in range(ng):
            if path[gg]:
                for ii in range(nk):
                    # Soft constraints
                    lambdac[ii][gg] = lambdav[offset2:offset2+ng][gg]
                    g += [gfcn(XD[ii][nicp-1][deg],XA[ii][nicp-1][deg-1],U[ii])[gg]-lambdac[ii][gg]]
                    lbg.append([-inf]*1)
                    ubg.append([0.]*1)
                    if gg == ng-1:
                        offset2 += ng
            else:
                ii = nk-1
                # Soft constraints
                lambdac[ii][gg] = lambdav[offset2:offset2+ng][gg]
                g += [gfcn(XD[ii][nicp-1][deg],XA[ii][nicp-1][deg-1],U[ii])[gg]-lambdac[ii][gg]]
                lbg.append([-inf]*1)
                ubg.append([0.]*1)
                if gg == ng-1:
                        offset2 += ng
            
        return vars_init, vars_lb, vars_ub, g, lbg, ubg, lambdav, XD, XA, U, cfcn, lambdac 

    def set_objective(self):
        lambdac, G, R    = self.lambdac, self.G, self.R
        nk, nicp, deg    = self.nk, self.nicp, self.deg
        U, XD, XA        = self.U, self.XD, self.XA
        nd, na, nu, nx   = self.nd, self.na, self.nu, self.nd+self.na
        ng, con, Obj_L, Obj_M = self.ng, self.con, self.Obj_L, self.Obj_M
        p_s              = con[nd:nd+nk]
        u_previous       = con[nd+nk:nd+nk+nu]
        Obj              = MX.zeros(1)
        
        # Soft-constraints for nonlinear constraints
        lg   = SX.sym('lg',ng)
        ps   = SX.sym('ps')
        lfcn = Function('lfcn',[lg,ps],[mtimes(mtimes(transpose(lg),G),lg)*ps])
        for k in range(nk):
            Obj += lfcn(lambdac[k],p_s[k])
        
        # Control penality
        u1     = SX.sym('u1',nu)
        u2     = SX.sym('u2',nu)
        dufcn  = Function('dufcn',[u1,u2,ps],[mtimes(mtimes(transpose(u2-u1),R),u2-u1)*ps])
        deltau = MX.zeros(1)
        for k in range(nk-1):
            if k == 0:
                deltau += dufcn(u_previous,U[k],p_s[k])
            else:
                deltau += dufcn(U[k],U[k+1],p_s[k])
        Obj += deltau
    
        # Lagrange term of objective
        lagrange = MX.zeros(1)
        for k in range(nk): 
            lagrange += Obj_L(XD[k][nicp-1][deg],XA[k][nicp-1][deg-1],U[k])*p_s[k]
        Obj += lagrange    
            
        # Mayer term of objective
        Obj += Obj_M(XD[nk-1][nicp-1][deg],XA[nk-1][nicp-1][deg-1],U[-1])
        
        return Obj

    def create_solver(self):
       V, con, Obj, g, opts = self.V, self.con, self.Obj, self.g, self.opts  
       
       # Define NLP
       nlp = {'x':V, 'p':con, 'f':Obj, 'g':vertcat(*g)} 
            
       # Allocate an NLP solver
       solver = nlpsol("solver", "ipopt", nlp, opts)   
       
       return solver 
   
    def initialization(self):
        tf, deltat, nu, nd = self.tf, self.tf/self.nk, self.nu, self.nd
        number_of_repeats, na, ng  = self.number_of_repeats, self.na, self.ng
            
        time_loop = []
        U_pasts   = np.zeros((number_of_repeats,int(math.ceil(tf/deltat)),nu))
        Xd_pasts  = np.zeros((int(math.ceil(tf/deltat))*100+1,number_of_repeats,nd))
        Xd_pastse = np.zeros((int(math.ceil(tf/deltat))+1,number_of_repeats,nd))
        Xa_pasts  = np.zeros((int(math.ceil(tf/deltat))*100,number_of_repeats,na)) 
        Con_pasts = np.zeros((int(math.ceil(tf/deltat))*100,number_of_repeats,ng))
        t_past    = [0.]
        u_nmpc    = np.array([0.]*nu)
        time_loop = []
        
        return U_pasts, Xd_pasts, Xd_pastse, Xa_pasts, Con_pasts, u_nmpc, time_loop
    
    def initialization_loop(self):
              
        x_hat0         = self.xhat0
        Sigmahat0      = self.Sigmahat0
        x0             = self.x0
        lbg, ubg, ng   = self.lbg, self.ubg, self.ng
        vars_lb, vars_ub, vars_init = self.vars_lb, self.vars_ub, self.vars_init
        tf, deltat, nu, nd = self.tf, self.tf/self.nk, self.nu, self.nd
        number_of_repeats, na  = self.number_of_repeats, self.na
        
        arg = {} 
        arg["lbg"] = np.concatenate(lbg)
        arg["ubg"] = np.concatenate(ubg)
        arg["lbx"] = vars_lb
        arg["ubx"] = vars_ub
        arg["x0"]  =  vars_init
        
        t_past    = [0.]
        u_past    = []
        tk        = -1
        t0i       = 0. 
        tfi       = 0. 
        
        return arg, u_past, x_hat0, Sigmahat0, x0, t_past, tk, t0i, tfi
    
    def simulator(self,xd_previous,uNMPC,t0,tf,xu_real):
        xd, xa, u, ODEeq, Aeq = self.xd, self.xa, self.u, self.ODEeq, self.Aeq
        xu = self.xu

        ODE = []
        for i in range(self.nd):
            ODE = vertcat(ODE,substitute(ODEeq[i],vertcat(u,xu),vertcat(SX(uNMPC),SX(xu_real)))) 
        
        A = []
        for i in range(self.na):
            A   = vertcat(A,substitute(Aeq[i],vertcat(u,xu),vertcat(SX(uNMPC),SX(xu_real))))
        
        dae = {'x':xd, 'z':xa, 'ode':ODE, 'alg':A}        
        I = integrator('I', 'idas', dae, {'t0':t0, 'tf':tf, 'abstol':1e-10, \
        'reltol':1e-10})
        res = I(x0=xd_previous)
        xd_current = array(res['xf'])
        xa_current = array(res['zf'])
        
        return xd_current, xa_current
    
    def update_inputs(self,x_hat,tk,u_nmpc):
        nd, nk, nu = self.nd, self.nk, self.nu
        tk    += 1
        p      = np.zeros(nd+nk+nu)
        if self.shrinking_horizon:
            a = np.concatenate((np.ones(nk-tk),np.zeros(tk)))
        else:
            a = np.ones(nk)
        
        p[:nd]            = np.array(x_hat)
        p[nd:nk+nd]       = a
        p[nk+nd:nk+nd+nu] = u_nmpc
     
        return p, tk
    
    def collect_data(self,t_past,u_past,time_taken,start,end,t0i,u_nmpc):
        t_past += [t0i] 
        u_past += [u_nmpc]
        time_taken += [end-start]
        
        return t_past, u_past, time_taken
    
    def generate_data(self,Xd_pasts,Xa_pasts,Con_pasts,U_pasts,un,loopend,\
                      time_loop,u_past,xu_real):
        simulation_time  = self.simulation_time
        t_pasts          = [0]
        xds              = self.x0
        Xd_pasts[0,un,:] = xds
        t0is             = 0. # start time of integrator
        tfis             = 0. # end time of integrator
        l = 0
        time_loop += [loopend]
        deltat, nu       = self.tf/self.nk, self.nu
        
        for k in range(int(math.ceil(simulation_time/deltat))):
                for i in range(nu):
                    U_pasts[un][k][i] = u_past[k][i]
        
        for k in range(int(math.ceil(simulation_time/deltat))):
            for o in range(100):
                l += 1
                tfis += deltat/100
                if t0is >= simulation_time:
                    break
                xds, xas = self.simulator(xds,u_past[k],t0is,tfis,xu_real)
                Xd_pasts[l,un,:]    = xds[:,0]
                Xa_pasts[l-1,un,:]  = xas[:,0]
                Con_pasts[l-1,un,:] = np.array(self.gfcn(xds,xas,u_past[k])).flatten()
                t0is += deltat/100
                t_pasts += [t0is] 
        
        return Xd_pasts, Xa_pasts, Con_pasts, U_pasts, time_loop, t_pasts
    
    def plot_graphs(self,t_past,t_pasts,Xd_pasts,Xa_pasts,U_pasts,Xd_pastse,Con_pasts):
        states              = self.states
        algebraics          = self.algebraics
        inputs              = self.inputs
        number_of_repeats   = self.number_of_repeats
        nd, na, nu, ng      = self.nd, self.na, self.nu, self.ng
        simulation_time     = self.simulation_time
        for j in range(nd):
            plt.figure(j)
            plt.clf()
            for i in range(number_of_repeats):
                plt.plot(t_pasts,Xd_pasts[:,i,j],'-')
            plt.ylabel(states[j])
            plt.xlabel('time')
            plt.xlim([0,simulation_time])
            plt.title('Monte Carlo trajectories of ' 
                          + states[j])
     
        for j in range(na):
            plt.figure(nd+j)
            plt.clf()
            for i in range(number_of_repeats):
                plt.plot(t_pasts[1:],Xa_pasts[:,i,j],'-')
            plt.ylabel(algebraics[j])
            plt.xlabel('time')
            plt.xlim([0,simulation_time])
            plt.title('Monte Carlo trajectories of ' 
                          + algebraics[j])
        
        for k in range(nu):
            plt.figure(nd+na+k)
            t_pastp = np.sort(np.concatenate([t_past]*2))
            plt.clf()
            for j in range(number_of_repeats):
                u_pastpF = []
                for i in range(len(U_pasts[j])):
                    u_pastpF += [U_pasts[j][i][0]]*2
                plt.plot(t_pastp[1:-1],u_pastpF,'-')
            plt.ylabel(inputs[k])
            plt.xlabel('time')
            plt.xlim([0,simulation_time])
            plt.title('Monte Carlo trajectories of ' 
                          + inputs[k])
            
        for j in range(nd):
            plt.figure(nd+na+nu+j)
            plt.clf()
            for i in range(number_of_repeats):
                plt.plot(t_past,Xd_pastse[:,i,j],marker="o")
            plt.ylabel(states[j] + " state estimate")
            plt.xlabel('time')
            plt.xlim([0,simulation_time])
            plt.title('Monte Carlo trajectories of ' 
                          + states[j] + " state estimate")
        
        for j in range(ng):
            plt.figure(nd+na+nu+j)
            plt.clf()
            for i in range(number_of_repeats):
                plt.plot(t_pasts[1:],Con_pasts[:,i,j],'-')
            plt.ylabel('g'+str(j))
            plt.xlabel('time')
            plt.xlim([0,t_pasts[-1]])
            plt.title('Monte Carlo trajectories of g' + str(j))
        
        num = nd+na+nu
        for j in range(nd):
            plt.figure(num+j)
            plt.clf()
            plt.plot(t_pasts,Xd_pasts[:,-1,j],'-',label='State')
            plt.plot(t_past,Xd_pastse[:,-1,j],marker="o",label='State estimate')
            plt.ylabel(states[j])
            plt.xlabel('time')
            plt.xlim([0,simulation_time])
            plt.title('Trajectory of ' 
                          + states[j] + ' for last Monte Carlo run')
            plt.legend()

        for j in range(na):
            plt.figure(num+nd+j)
            plt.clf()
            plt.plot(t_pasts[1:],Xa_pasts[:,-1,j],'-')
            plt.ylabel(algebraics[j])
            plt.xlabel('time')
            plt.xlim([0,simulation_time])
            plt.title('Trajectory of ' 
                          + algebraics[j] + ' for last Monte Carlo run')
        
        for k in range(nu):
            plt.figure(num+nd+na+k)
            t_pastp = np.sort(np.concatenate([t_past]*2))
            plt.clf()
            u_pastpF = []
            for i in range(len(U_pasts[-1])):
                u_pastpF += [U_pasts[-1][i][0]]*2
            plt.plot(t_pastp[1:-1],u_pastpF,'-')
            plt.ylabel(inputs[k])
            plt.xlabel('time')
            plt.xlim([0,simulation_time])  
            plt.title('Trajectory of ' 
                          + inputs[k] + ' for last Monte Carlo run')
        
        for j in range(ng):
            plt.figure(num+nd+na+nu+j)
            plt.clf()
            for i in range(number_of_repeats):
                plt.plot(t_pasts[1:],Con_pasts[:,i,j],'-')
            plt.ylabel('g'+str(j))
            plt.xlabel('time')
            plt.xlim([0,t_pasts[-1]])
            plt.title('Trajectory of g' 
                      + str(j) + ' for last Monte Carlo run')
                    
        Data_NMPC                                = {}
        Data_NMPC['differential_states']         = Xd_pasts
        Data_NMPC['algebraic_states']            = Xa_pasts
        Data_NMPC['inputs']                      = U_pasts
        Data_NMPC['constraints']                 = Con_pasts
        Data_NMPC['simulation_times']            = t_pasts
        Data_NMPC['state_estimates']             = Xd_pastse
        savemat('Data_NMPC',Data_NMPC)    
            
        return
    
    def load_varsopthyp(self):
        try:
            with open("varsopt" + ".pkl", 'rb') as a_file:
                vars_init = pickle.load(a_file)
        except:
            print("error loading varsopt")
        return vars_init

    def save_varsopthyp(self,varsopt):
        try:
            with open("varspopt" + ".pkl", 'wb') as a_file:
                pickle.dump(varsopt,a_file)
        except:
            print("error saving varsopt")
        return