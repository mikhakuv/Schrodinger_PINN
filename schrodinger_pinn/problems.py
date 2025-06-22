import math
import torch
import numpy as np
import matplotlib.pyplot as plt

class problem():
    def show_ic(self, x_resolution=200):
        x = np.linspace(self.x_0, self.x_1, x_resolution)
        t = self.t_0*np.ones(x_resolution)
        Q = self.q(x,t)
        U=np.real(Q)
        V=np.imag(Q)
        Q_abs = np.abs(Q)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17,5))
        ax1.plot(x, Q_abs, color = "cornflowerblue")
        ax1.set_title(f'|q(x, {self.t_0})|')
        ax2.plot(x, U, color = "cornflowerblue")
        ax2.set_title(f'Re(q(x, {self.t_0}))')
        ax3.plot(x, V, color = "cornflowerblue")
        ax3.set_title(f'Im(q(x, {self.t_0}))')
        plt.show()
        return
        
    def show_solution(self, x_resolution=200, t_resolution=100, show_residual=True):
        x = np.linspace(self.x_0, self.x_1, x_resolution)
        t = np.linspace(self.t_0, self.t_1, t_resolution)
        X, T = np.meshgrid(x, t)
        Q = self.q(X,T)
        U=np.real(Q)
        V=np.imag(Q)
        Q_abs = np.abs(Q)

        ##plot solution
        fig, axs = plt.subplots(1, 2, figsize=(12,5), dpi=100)
        axs[0].set(xlabel='$t$', ylabel='$x$')
        axs[0].set_title('$Re(Q)$')
        c = axs[0].pcolormesh(T, X, U, shading='nearest', cmap='RdBu')
        axs[0].axis([self.t_0, self.t_1, self.x_0, self.x_1])
        fig.colorbar(c, ax=axs[0])
        axs[1].set(xlabel='$t$', ylabel='$x$')
        axs[1].set_title('$Im(Q)$')
        c = axs[1].pcolormesh(T, X, V, shading='nearest', cmap='RdBu')
        axs[1].axis([self.t_0, self.t_1, self.x_0, self.x_1])
        fig.colorbar(c, ax=axs[1])

        if show_residual:
            ##plot residual of solution
            X_tensor = torch.tensor(X, requires_grad=True)
            T_tensor = torch.tensor(T, requires_grad=True)
            Q = self.q_tensor(X_tensor, T_tensor)
            u = torch.real(Q)
            v = torch.imag(Q)
            Re, Im = self.calc_res(u, v, X_tensor, T_tensor)
            Residual = (Re**2 + Im**2)**0.5
            fig, ax = plt.subplots(1, 1, figsize=(6,5), dpi=100)
            ax.set(xlabel='$t$', ylabel='$x$')
            ax.set_title('$|Res(Q)|$')
            c = ax.pcolormesh(T, X, Residual.detach().numpy(), shading='nearest', cmap='Reds')
            ax.axis([self.t_0, self.t_1, self.x_0, self.x_1])
            fig.colorbar(c, ax=ax)
        plt.show()
        return

    def find_half_width(self, eps=0.001):
        for partition in np.logspace(2,7,10):
            X_arr = np.linspace(self.x_0,self.x_1,int(partition))
            F_arr = np.abs(self.q(X_arr,self.t_0))
            max_height = np.max(F_arr)
            half_max_height = max_height/2
            diffs = np.abs(F_arr-half_max_height)
            intersections = X_arr[diffs<=eps]
            if len(intersections)>1:
              plt.figure(figsize=(5,5))
              plt.title(f'|q(x, {self.t_0})|')
              plt.plot(X_arr,F_arr,color="cornflowerblue")
              plt.plot(X_arr,max_height*np.ones_like(X_arr),"--r")
              plt.plot(X_arr,half_max_height*np.ones_like(X_arr),"--r")
              plt.scatter([np.min(intersections),np.max(intersections)], [np.abs(self.q(np.min(intersections), self.t_0)),np.abs(self.q(np.max(intersections), self.t_0))], color="red")
              plt.xlim(self.x_0, self.x_1)
              plt.ylim(0,max_height*1.05)
              plt.show()
              return (np.max(intersections) - np.min(intersections)).item()
        print("Halfwidth not found")
        return -1

class semilinear(problem):
    def __init__(self, x_0, x_1, t_0, t_1):
        #domain settings
        self.x_0 = x_0
        self.x_1 = x_1
        self.t_0 = t_0
        self.t_1 = t_1
    
    #analytical solution:
    def q(self,x,t):
        return 2/np.cos(1j*x + 0*t)
    def q_tensor(self,x,t):
        return 2/torch.cos(1j*x + 0*t)

    #residual calculation
    def calc_res(self, u, v, X_tensor, T_tensor):
        #real part of derivatives
        u_t = torch.autograd.grad(u,T_tensor,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_x = torch.autograd.grad(u,X_tensor,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x,X_tensor,grad_outputs=torch.ones_like(u_x),retain_graph=True,create_graph=True)[0]
        #complex part of derivatives
        v_t = torch.autograd.grad(v,T_tensor,grad_outputs=torch.ones_like(v),retain_graph=True,create_graph=True)[0]
        v_x = torch.autograd.grad(v,X_tensor,grad_outputs=torch.ones_like(v),retain_graph=True,create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x,X_tensor,grad_outputs=torch.ones_like(v_x),retain_graph=True,create_graph=True)[0]
        
        #i*q_t + 0.5*q_xx + q*|q|**2 = 0
        Real_res  =  -v_t + u*(u**2 + v**2) + 0.5*u_xx
        Complex_res = u_t + v*(u**2 + v**2) + 0.5*v_xx
        return Real_res, Complex_res
    
    #boundary condition calculation
    def bound_cond(self, u, v, X_tensor, T_tensor):
        #real part of derivatives
        u_x = torch.autograd.grad(u,X_tensor,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        #complex part of derivatives
        v_x = torch.autograd.grad(v,X_tensor,grad_outputs=torch.ones_like(v),retain_graph=True,create_graph=True)[0]
        #periodic boundary conditions
        loss_b = torch.mean(((u[X_tensor==self.x_0]-u[X_tensor==self.x_1])**2 + (v[X_tensor==self.x_0]-v[X_tensor==self.x_1])**2 +\
                             (u_x[X_tensor==self.x_0]-u_x[X_tensor==self.x_1])**2 + (v_x[X_tensor==self.x_0]-v_x[X_tensor==self.x_1])**2)/4)
        return loss_b
    
    #info functions:
    def show_params(self): #recalculate dependent parameters and shows them
        print(f"no parameters in this equation")
        return

class second_order(problem):
    def __init__(self, x_0, x_1, t_0, t_1, alpha, beta, k_param, w_param, x0_param, theta0):
        #domain settings
        self.x_0 = x_0
        self.x_1 = x_1
        self.t_0 = t_0
        self.t_1 = t_1
        #equation and initial condition parameters:
        self.alpha = alpha
        self.beta = beta #given solution satisfies equation only if beta=0
        self.k = k_param
        self.w = w_param
        self.x0_param = x0_param
        self.th0 = theta0
    
    #analytical solution:
    def q(self,x,t):
      mu = 4*(self.k**2 - self.w)
      nu = 4*self.alpha/3
      kexp = np.exp((mu**0.5)*(x - 2*self.k*t - self.x0_param))
      f_com = ((4*mu*kexp)/(1 + 4*kexp + 4*(1+mu*nu)*kexp**2))**0.5
      u = f_com*np.cos(self.k*x - self.w*t + self.th0)
      v = f_com*np.sin(self.k*x - self.w*t + self.th0)
      return u+1j*v
    def q_tensor(self,x,t):
      mu = 4*(self.k**2 - self.w)
      nu = 4*self.alpha/3
      kexp = torch.exp((mu**0.5)*(x - 2*self.k*t - self.x0_param))
      f_com = ((4*mu*kexp)/(1 + 4*kexp + 4*(1+mu*nu)*kexp**2))**0.5
      u = f_com*torch.cos(self.k*x - self.w*t + self.th0)
      v = f_com*torch.sin(self.k*x - self.w*t + self.th0)
      return u+1j*v

    #residual calculation
    def calc_res(self, u, v, X_tensor, T_tensor):
        #real part of derivatives
        u_t = torch.autograd.grad(u,T_tensor,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_x = torch.autograd.grad(u,X_tensor,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x,X_tensor,grad_outputs=torch.ones_like(u_x),retain_graph=True,create_graph=True)[0]
        #complex part of derivatives
        v_t = torch.autograd.grad(v,T_tensor,grad_outputs=torch.ones_like(v),retain_graph=True,create_graph=True)[0]
        v_x = torch.autograd.grad(v,X_tensor,grad_outputs=torch.ones_like(v),retain_graph=True,create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x,X_tensor,grad_outputs=torch.ones_like(v_x),retain_graph=True,create_graph=True)[0]
        
        #i*q_t + q_xx + q*(|q|**2 + alpha*|q|**4 + beta*|q|**6) = 0
        Real_res  =  -v_t + u*(u**2 + v**2)*(1 + self.alpha*(u**2 + v**2) + self.beta*(u**2 + v**2)**2) + u_xx
        Complex_res = u_t + v*(u**2 + v**2)*(1 + self.alpha*(u**2 + v**2) + self.beta*(u**2 + v**2)**2) + v_xx
        return Real_res, Complex_res

    #boundary conditions calculation
    def bound_cond(self, u, v, X_tensor, T_tensor):
        #Dirichlet boundary conditions
        loss_b = torch.mean((u**2 + v**2)/2)
        return loss_b
    
    #info functions:
    def show_params(self): #recalculate dependent parameters and shows them
        print(f"alpha = {self.alpha:.3f}, beta = {self.beta:.3f}, k = {self.k:.3f}, w = {self.w:.3f}, x0 = {self.x0_param:.3f}, th0 = {self.th0:.3f}")
        return

class fourth_order(problem):
    def __init__(self, x_0, x_1, t_0, t_1, a1, a2, a4, A_param, k_param, nu, mu, x0_param, theta0):
        #domain settings
        self.x_0 = x_0
        self.x_1 = x_1
        self.t_0 = t_0
        self.t_1 = t_1
        #equation and initial condition parameters:
        self.a1 = a1
        self.a2 = a2
        self.a4 = a4
        self.A = A_param
        self.k = k_param
        self.nu = nu
        self.mu = mu
        self.x0_param = x0_param
        self.th0 = theta0
        #dependent parameters
        self.khi = 0
        self.v = 8*(self.k**3)*self.a4 + 2*self.k*self.a2 + self.a1
        self.w = (3*self.k**4 - 6*self.nu*(self.k**2) - 12*self.khi*self.mu - self.nu**2)*self.a4 + (self.k**2-self.nu)*self.a2 + self.k*self.a1
        self.a3 = -4*self.k*self.a4
        self.b1 = ((12*self.mu*self.k**2 + 20*self.mu*self.nu + 60*self.khi)*self.a4 + 2*self.mu*self.a2)/self.A
        self.b2 = ((18*self.k**2 + 24*self.mu**2 + 78*self.nu)*self.a4 + 3*self.a2)/(self.A**2)
        self.b3 = 120*self.mu*self.a4/(self.A**3)
        self.b4 = 105*self.a4/(self.A**4)
    
    #analytical solution:
    def q(self,x,t):
      f_com = ((2*self.A*self.nu)**0.5)/(-self.mu+np.cosh(2*(x-self.v*t-self.x0_param)*self.nu**0.5)*(self.mu**2-4*self.nu)**0.5)**0.5
      u = f_com*np.cos(self.k*x - self.w*t + self.th0)
      v = f_com*np.sin(self.k*x - self.w*t + self.th0)
      return u+1j*v
    def q_tensor(self,x,t):
      f_com = ((2*self.A*self.nu)**0.5)/(-self.mu+torch.cosh(2*(x-self.v*t-self.x0_param)*self.nu**0.5)*(self.mu**2-4*self.nu)**0.5)**0.5
      u = f_com*torch.cos(self.k*x - self.w*t + self.th0)
      v = f_com*torch.sin(self.k*x - self.w*t + self.th0)
      return u+1j*v

    #residual calculation
    def calc_res(self, u, v, X_tensor, T_tensor):
        #real part of derivatives
        u_t = torch.autograd.grad(u,T_tensor,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_x = torch.autograd.grad(u,X_tensor,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x,X_tensor,grad_outputs=torch.ones_like(u_x),retain_graph=True,create_graph=True)[0]
        u_3x = torch.autograd.grad(u_xx,X_tensor,grad_outputs=torch.ones_like(u_xx),retain_graph=True,create_graph=True)[0]
        u_4x = torch.autograd.grad(u_3x,X_tensor,grad_outputs=torch.ones_like(u_3x),retain_graph=True,create_graph=True)[0]
        #complex part of derivatives
        v_t = torch.autograd.grad(v,T_tensor,grad_outputs=torch.ones_like(v),retain_graph=True,create_graph=True)[0]
        v_x = torch.autograd.grad(v,X_tensor,grad_outputs=torch.ones_like(v),retain_graph=True,create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x,X_tensor,grad_outputs=torch.ones_like(v_x),retain_graph=True,create_graph=True)[0]
        v_3x = torch.autograd.grad(v_xx,X_tensor,grad_outputs=torch.ones_like(v_xx),retain_graph=True,create_graph=True)[0]
        v_4x = torch.autograd.grad(v_3x,X_tensor,grad_outputs=torch.ones_like(v_3x),retain_graph=True,create_graph=True)[0]
        
        #i*q_t + i*a_1*q_x + a_2*q_xx + i*a_3*q_3x + a_4*q_4x - q*(b_1*|q|**2 + b_2*|q|**4 + b_3*|q|**6 + b_4*|q|**8) = 0
        Real_res  =  -v_t - self.a1*v_x + self.a2*u_xx - self.a3*v_3x + self.a4*u_4x - u*(u**2 + v**2)*(self.b1 + self.b2*(u**2 + v**2) + self.b3*(u**2 + v**2)**2 + self.b4*(u**2 + v**2)**3)
        Complex_res = u_t + self.a1*u_x + self.a2*v_xx + self.a3*u_3x + self.a4*v_4x - v*(u**2 + v**2)*(self.b1 + self.b2*(u**2 + v**2) + self.b3*(u**2 + v**2)**2 + self.b4*(u**2 + v**2)**3)
        return Real_res, Complex_res

    #boundary conditions calculation
    def bound_cond(self, u, v, X_tensor, T_tensor):
        #real part of derivatives
        u_x = torch.autograd.grad(u,X_tensor,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x,X_tensor,grad_outputs=torch.ones_like(u_x),retain_graph=True,create_graph=True)[0]
        u_3x = torch.autograd.grad(u_xx,X_tensor,grad_outputs=torch.ones_like(u_xx),retain_graph=True,create_graph=True)[0]
        #complex part of derivatives
        v_x = torch.autograd.grad(v,X_tensor,grad_outputs=torch.ones_like(v),retain_graph=True,create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x,X_tensor,grad_outputs=torch.ones_like(v_x),retain_graph=True,create_graph=True)[0]
        v_3x = torch.autograd.grad(v_xx,X_tensor,grad_outputs=torch.ones_like(v_xx),retain_graph=True,create_graph=True)[0]
        #Periodic boundary conditions
        loss_b = torch.mean(((u[X_tensor==self.x_0]-u[X_tensor==self.x_1])**2 + (v[X_tensor==self.x_0]-v[X_tensor==self.x_1])**2 +\
                             (u_x[X_tensor==self.x_0]-u_x[X_tensor==self.x_1])**2 + (v_x[X_tensor==self.x_0]-v_x[X_tensor==self.x_1])**2 +\
                             (u_xx[X_tensor==self.x_0]-u_xx[X_tensor==self.x_1])**2 + (v_xx[X_tensor==self.x_0]-v_xx[X_tensor==self.x_1])**2 +\
                             (u_3x[X_tensor==self.x_0]-u_3x[X_tensor==self.x_1])**2 + (v_3x[X_tensor==self.x_0]-v_3x[X_tensor==self.x_1])**2)/8)
        return loss_b
    
    #info functions:
    def show_params(self): #recalculate dependent parameters and shows them
        print(f"a1 = {self.a1:.3f}, a2 = {self.a2:.3f}, a3 = {self.a3:.3f}, a4 = {self.a4:.3f},\n b1 = {self.b1:.3f}, b2 = {self.b2:.3f}, b3 = {self.b3:.3f}, b4 = {self.b4:.3f}")
        return

class sixth_order(problem):
    def __init__(self, x_0, x_1, t_0, t_1, a1, a2, a4, a6, b1, khi, a_param, x0_param, theta0):
        #domain settings
        self.x_0 = x_0
        self.x_1 = x_1
        self.t_0 = t_0
        self.t_1 = t_1
        #equation and initial condition parameters:
        self.a1 = a1
        self.a2 = a2
        self.a4 = a4
        self.a6 = a6
        self.b1 = b1
        self.k_param = 1 #k_param must be 1 or 0
        self.khi = khi
        self.a_param = a_param
        self.x0_param = x0_param
        self.theta0 = theta0
        #dependent parameters
        self.A1_param = (2*self.khi*(self.a2-6*self.a4*(self.k_param**2) + 12*self.a4*self.k_param + 10*self.a4 + 75*self.a6*(self.k_param**4) + 150*self.a6*(self.k_param**2) + 91*self.a6)/self.b1)**0.5
        self.C0_param = self.a1 + 2*self.a2*self.k_param + 8*self.a4*(self.k_param**3) + 96*self.a6*(self.k_param**5)
        self.w_param = self.a1*self.k_param + self.a2*(self.k_param**2) - self.a2 + 3*self.a4*(self.k_param**4) - 6*self.a4*(self.k_param**2) - self.a4 + 35*self.a6*(self.k_param**6) - 75*self.a6*(self.k_param**4) - 15*self.a6*(self.k_param**2) - self.a6
        self.a3 = -4*self.a4*self.k_param - 40*self.a6*(self.k_param**3)
        self.a5 = -6*self.a6*self.k_param
        self.b2 = -(24*self.a4*(self.khi**2) + 360*self.a6*(self.khi**2)*(self.k_param**2) + 840*self.a6*(self.khi**2))/(self.A1_param**4)
        self.b3 = 720*self.a6*(self.khi**3)/(self.A1_param**6)
    
    #analytical solution:
    def q(self,x,t):
        frac = self.A1_param/(self.a_param*np.exp(x-self.C0_param*t+self.x0_param) + self.khi/(4*self.a_param*np.exp(x-self.C0_param*t+self.x0_param)))
        u = frac*np.cos(self.k_param*x-self.w_param*t+self.theta0)
        v = frac*np.sin(self.k_param*x-self.w_param*t+self.theta0)
        return u+1j*v
    def q_tensor(self,x,t):
        frac = self.A1_param/(self.a_param*torch.exp(x-self.C0_param*t+self.x0_param) + self.khi/(4*self.a_param*torch.exp(x-self.C0_param*t+self.x0_param)))
        u = frac*torch.cos(self.k_param*x-self.w_param*t+self.theta0)
        v = frac*torch.sin(self.k_param*x-self.w_param*t+self.theta0)
        return u+1j*v

    #residual calculation
    def calc_res(self, u, v, X_tensor, T_tensor):
        #real part of derivatives
        u_t = torch.autograd.grad(u,T_tensor,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_x = torch.autograd.grad(u,X_tensor,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x,X_tensor,grad_outputs=torch.ones_like(u_x),retain_graph=True,create_graph=True)[0]
        u_3x = torch.autograd.grad(u_xx,X_tensor,grad_outputs=torch.ones_like(u_xx),retain_graph=True,create_graph=True)[0]
        u_4x = torch.autograd.grad(u_3x,X_tensor,grad_outputs=torch.ones_like(u_3x),retain_graph=True,create_graph=True)[0]
        u_5x = torch.autograd.grad(u_4x,X_tensor,grad_outputs=torch.ones_like(u_4x),retain_graph=True,create_graph=True)[0]
        u_6x = torch.autograd.grad(u_5x,X_tensor,grad_outputs=torch.ones_like(u_5x),retain_graph=True,create_graph=True)[0]
        #complex part of derivatives
        v_t = torch.autograd.grad(v,T_tensor,grad_outputs=torch.ones_like(v),retain_graph=True,create_graph=True)[0]
        v_x = torch.autograd.grad(v,X_tensor,grad_outputs=torch.ones_like(v),retain_graph=True,create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x,X_tensor,grad_outputs=torch.ones_like(v_x),retain_graph=True,create_graph=True)[0]
        v_3x = torch.autograd.grad(v_xx,X_tensor,grad_outputs=torch.ones_like(v_xx),retain_graph=True,create_graph=True)[0]
        v_4x = torch.autograd.grad(v_3x,X_tensor,grad_outputs=torch.ones_like(v_3x),retain_graph=True,create_graph=True)[0]
        v_5x = torch.autograd.grad(v_4x,X_tensor,grad_outputs=torch.ones_like(v_4x),retain_graph=True,create_graph=True)[0]
        v_6x = torch.autograd.grad(v_5x,X_tensor,grad_outputs=torch.ones_like(v_5x),retain_graph=True,create_graph=True)[0]
        
        #i*q_t + i*a1*q_x + a2*q_xx + i*a3*q_3x + a4*q_4x + i*a5*q_5x + a6*q_6x + q*(b1*|q|**2 + b2*|q|**4 + b3*|q|**6) = 0
        Real_res = -v_t - self.a1*v_x + self.a2*u_xx - self.a3*v_3x + self.a4*u_4x - self.a5*v_5x + self.a6*u_6x + u*(self.b1*(u**2 + v**2) + self.b2*(u**2 + v**2)**2 + self.b3*(u**2 + v**2)**3)
        Complex_res = u_t + self.a1*u_x + self.a2*v_xx + self.a3*u_3x + self.a4*v_4x + self.a5*u_5x + self.a6*v_6x + v*(self.b1*(u**2 + v**2) + self.b2*(u**2 + v**2)**2 + self.b3*(u**2 + v**2)**3)
        return Real_res, Complex_res

    #boundary conditions calculation
    def bound_cond(self, u, v, X_tensor, T_tensor):
        #real part of derivatives
        u_x = torch.autograd.grad(u,X_tensor,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x,X_tensor,grad_outputs=torch.ones_like(u_x),retain_graph=True,create_graph=True)[0]
        #complex part of derivatives
        v_x = torch.autograd.grad(v,X_tensor,grad_outputs=torch.ones_like(v),retain_graph=True,create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x,X_tensor,grad_outputs=torch.ones_like(v_x),retain_graph=True,create_graph=True)[0]
        #Dirichlet boundary conditions
        loss_b = torch.mean((u**2 + u_x**2 + u_xx**2 + v**2 + v_x**2 + v_xx**2)/6)
        return loss_b
    
    #info functions:
    def show_params(self):
        print(f"a1 = {self.a1:.3f}, a2 = {self.a2:.3f}, a3 = {self.a3:.3f}, a4 = {self.a4:.3f}, a5 = {self.a5:.3f}, a6 = {self.a6:.3f},\n b1 = {self.b1:.3f}, b2 = {self.b2:.3f}, b3 = {self.b3:.3f}")
        return

class sixth_order_custom(sixth_order):
    #change only initial condition:
    def q(self,x,t):
        frac = self.A1_param/(self.a_param*np.exp(x-self.C0_param*t+self.x0_param) + self.khi/(4*self.a_param*np.exp(x-self.C0_param*t+self.x0_param)))
        u = frac*np.cos(self.k_param*x-self.w_param*t+self.theta0)
        v = frac*np.sin(self.k_param*x-self.w_param*t+self.theta0)
        return u+1j*v + 2*np.exp(-(x-1)**2)
    def q_tensor(self,x,t):
        frac = self.A1_param/(self.a_param*torch.exp(x-self.C0_param*t+self.x0_param) + self.khi/(4*self.a_param*torch.exp(x-self.C0_param*t+self.x0_param)))
        u = frac*torch.cos(self.k_param*x-self.w_param*t+self.theta0)
        v = frac*torch.sin(self.k_param*x-self.w_param*t+self.theta0)
        return u+1j*v + 2*torch.exp(-(x-1)**2)

class sixth_order_two_solitons(sixth_order):
    def q(self,x,t):
        distance = 2.63566*4
        #first soliton: solution
        x0_param_1 = self.x0_param
        frac_1 = self.A1_param/(self.a_param*np.exp(x-self.C0_param*t+x0_param_1) + self.khi/(4*self.a_param*np.exp(x-self.C0_param*t+x0_param_1)))
        u_1 = frac_1*np.cos(self.k_param*x-self.w_param*t+self.theta0)
        v_1 = frac_1*np.sin(self.k_param*x-self.w_param*t+self.theta0)
        #second soliton: starts before first
        x0_param_2 = self.x0_param + distance
        frac_2 = self.A1_param/(self.a_param*np.exp(x-self.C0_param*t+x0_param_2) + self.khi/(4*self.a_param*np.exp(x-self.C0_param*t+x0_param_2)))
        u_2 = frac_2*np.cos(self.k_param*x-self.w_param*t+self.theta0)
        v_2 = frac_2*np.sin(self.k_param*x-self.w_param*t+self.theta0)
        return (u_1+u_2) + 1j*(v_1+v_2)
    def q_tensor(self,x,t):
        distance = 2.63566*4
        #first soliton: solution
        x0_param_1 = self.x0_param
        frac_1 = self.A1_param/(self.a_param*torch.exp(x-self.C0_param*t+x0_param_1) + self.khi/(4*self.a_param*torch.exp(x-self.C0_param*t+x0_param_1)))
        u_1 = frac_1*torch.cos(self.k_param*x-self.w_param*t+self.theta0)
        v_1 = frac_1*torch.sin(self.k_param*x-self.w_param*t+self.theta0)
        #second soliton: starts before first
        x0_param_2 = self.x0_param + distance
        frac_2 = self.A1_param/(self.a_param*torch.exp(x-self.C0_param*t+x0_param_2) + self.khi/(4*self.a_param*torch.exp(x-self.C0_param*t+x0_param_2)))
        u_2 = frac_2*torch.cos(self.k_param*x-self.w_param*t+self.theta0)
        v_2 = frac_2*torch.sin(self.k_param*x-self.w_param*t+self.theta0)
        return (u_1+u_2) + 1j*(v_1+v_2)

class sixth_order_three_solitons(sixth_order):
    def q(self,x,t):
        distance = 2.63566*4
        #first soliton: solution
        x0_param_1 = self.x0_param
        frac_1 = self.A1_param/(self.a_param*np.exp(x-self.C0_param*t+x0_param_1) + self.khi/(4*self.a_param*np.exp(x-self.C0_param*t+x0_param_1)))
        u_1 = frac_1*np.cos(self.k_param*x-self.w_param*t+self.theta0)
        v_1 = frac_1*np.sin(self.k_param*x-self.w_param*t+self.theta0)
        #second soliton: starts before first
        x0_param_2 = self.x0_param + distance
        frac_2 = self.A1_param/(self.a_param*np.exp(x-self.C0_param*t+x0_param_2) + self.khi/(4*self.a_param*np.exp(x-self.C0_param*t+x0_param_2)))
        u_2 = frac_2*np.cos(self.k_param*x-self.w_param*t+self.theta0)
        v_2 = frac_2*np.sin(self.k_param*x-self.w_param*t+self.theta0)
        #third soliton: starts before second
        x0_param_3 = self.x0_param + 2*distance
        frac_3 = self.A1_param/(self.a_param*np.exp(x-self.C0_param*t+x0_param_3) + self.khi/(4*self.a_param*np.exp(x-self.C0_param*t+x0_param_3)))
        u_3 = frac_3*np.cos(self.k_param*x-self.w_param*t+self.theta0)
        v_3 = frac_3*np.sin(self.k_param*x-self.w_param*t+self.theta0)
        return (u_1+u_2+u_3) + 1j*(v_1+v_2+v_3)
    def q_tensor(self,x,t):
        distance = 2.63566*4
        #first soliton: solution
        x0_param_1 = self.x0_param
        frac_1 = self.A1_param/(self.a_param*torch.exp(x-self.C0_param*t+x0_param_1) + self.khi/(4*self.a_param*torch.exp(x-self.C0_param*t+x0_param_1)))
        u_1 = frac_1*torch.cos(self.k_param*x-self.w_param*t+self.theta0)
        v_1 = frac_1*torch.sin(self.k_param*x-self.w_param*t+self.theta0)
        #second soliton: starts before first
        x0_param_2 = self.x0_param + distance
        frac_2 = self.A1_param/(self.a_param*torch.exp(x-self.C0_param*t+x0_param_2) + self.khi/(4*self.a_param*torch.exp(x-self.C0_param*t+x0_param_2)))
        u_2 = frac_2*torch.cos(self.k_param*x-self.w_param*t+self.theta0)
        v_2 = frac_2*torch.sin(self.k_param*x-self.w_param*t+self.theta0)
        #third soliton: starts before second
        x0_param_3 = self.x0_param + 2*distance
        frac_3 = self.A1_param/(self.a_param*torch.exp(x-self.C0_param*t+x0_param_3) + self.khi/(4*self.a_param*torch.exp(x-self.C0_param*t+x0_param_3)))
        u_3 = frac_3*torch.cos(self.k_param*x-self.w_param*t+self.theta0)
        v_3 = frac_3*torch.sin(self.k_param*x-self.w_param*t+self.theta0)
        return (u_1+u_2+u_3) + 1j*(v_1+v_2+v_3)

class sixth_order_solitons_colision(sixth_order):
    def q(self,x,t):
        #first soliton: solution
        k_param_1 = self.k_param
        x0_param_1 = self.x0_param
        frac_1 = self.A1_param/(self.a_param*np.exp(x-self.C0_param*t+x0_param_1) + self.khi/(4*self.a_param*np.exp(x-self.C0_param*t+x0_param_1)))
        u_1 = frac_1*np.cos(k_param_1*x-self.w_param*t+self.theta0)
        v_1 = frac_1*np.sin(k_param_1*x-self.w_param*t+self.theta0)
        #second soliton: goes first, spoiled solution
        k_param_2 = 0
        A1_param = (2*self.khi*(self.a2-6*self.a4*(k_param_2**2) + 12*self.a4*k_param_2 + 10*self.a4 + 75*self.a6*(k_param_2**4) + 150*self.a6*(k_param_2**2) + 91*self.a6)/self.b1)**0.5
        C0_param = self.a1 + 2*self.a2*k_param_2 + 8*self.a4*(k_param_2**3) + 96*self.a6*(k_param_2**5)
        w_param = self.a1*k_param_2 + self.a2*(k_param_2**2) - self.a2 + 3*self.a4*(k_param_2**4) - 6*self.a4*(k_param_2**2) - self.a4 + 35*self.a6*(k_param_2**6) - 75*self.a6*(k_param_2**4) - 15*self.a6*(k_param_2**2) - self.a6
        x0_param_2 = self.x0_param - 10
        frac_2 = A1_param/(self.a_param*np.exp(x-C0_param*t+x0_param_2) + self.khi/(4*self.a_param*np.exp(x-C0_param*t+x0_param_2)))
        u_2 = frac_2*np.cos(k_param_2*x-w_param*t+self.theta0)
        v_2 = frac_2*np.sin(k_param_2*x-w_param*t+self.theta0)
        return (u_1+u_2) + 1j*(v_1+v_2)
    def q_tensor(self,x,t):
        #first soliton: solution
        k_param_1 = self.k_param
        x0_param_1 = self.x0_param
        frac_1 = self.A1_param/(self.a_param*torch.exp(x-self.C0_param*t+x0_param_1) + self.khi/(4*self.a_param*torch.exp(x-self.C0_param*t+x0_param_1)))
        u_1 = frac_1*torch.cos(k_param_1*x-self.w_param*t+self.theta0)
        v_1 = frac_1*torch.sin(k_param_1*x-self.w_param*t+self.theta0)
        #second soliton: goes first, spoiled solution
        k_param_2 = 0
        A1_param = (2*self.khi*(self.a2-6*self.a4*(k_param_2**2) + 12*self.a4*k_param_2 + 10*self.a4 + 75*self.a6*(k_param_2**4) + 150*self.a6*(k_param_2**2) + 91*self.a6)/self.b1)**0.5
        C0_param = self.a1 + 2*self.a2*k_param_2 + 8*self.a4*(k_param_2**3) + 96*self.a6*(k_param_2**5)
        w_param = self.a1*k_param_2 + self.a2*(k_param_2**2) - self.a2 + 3*self.a4*(k_param_2**4) - 6*self.a4*(k_param_2**2) - self.a4 + 35*self.a6*(k_param_2**6) - 75*self.a6*(k_param_2**4) - 15*self.a6*(k_param_2**2) - self.a6
        x0_param_2 = self.x0_param - 10
        frac_2 = A1_param/(self.a_param*torch.exp(x-C0_param*t+x0_param_2) + self.khi/(4*self.a_param*torch.exp(C0_param*t+x0_param_2)))
        u_2 = frac_2*torch.cos(k_param_2*x-w_param*t+self.theta0)
        v_2 = frac_2*torch.sin(k_param_2*x-w_param*t+self.theta0)
        return (u_1+u_2) + 1j*(v_1+v_2)

class sixth_order_strange_soliton(sixth_order):
    #soliton is called strange because all coefficients in equation are calculated for k=1 while the soliton has k=0
    def q(self,x,t):
        k_param = 0
        A1_param = (2*self.khi*(self.a2-6*self.a4*(k_param**2) + 12*self.a4*k_param + 10*self.a4 + 75*self.a6*(k_param**4) + 150*self.a6*(k_param**2) + 91*self.a6)/self.b1)**0.5
        C0_param = self.a1 + 2*self.a2*k_param + 8*self.a4*(k_param**3) + 96*self.a6*(k_param**5)
        w_param = self.a1*k_param + self.a2*(k_param**2) - self.a2 + 3*self.a4*(k_param**4) - 6*self.a4*(k_param**2) - self.a4 + 35*self.a6*(k_param**6) - 75*self.a6*(k_param**4) - 15*self.a6*(k_param**2) - self.a6
        x0_param = self.x0_param
        frac = A1_param/(self.a_param*np.exp(x-C0_param*t+x0_param) + self.khi/(4*self.a_param*np.exp(x-C0_param*t+x0_param)))
        u = frac*np.cos(k_param*x-w_param*t+self.theta0)
        v = frac*np.sin(k_param*x-w_param*t+self.theta0)
        return u + 1j*v
    def q_tensor(self,x,t):
        k_param = 0
        A1_param = (2*self.khi*(self.a2-6*self.a4*(k_param**2) + 12*self.a4*k_param + 10*self.a4 + 75*self.a6*(k_param**4) + 150*self.a6*(k_param**2) + 91*self.a6)/self.b1)**0.5
        C0_param = self.a1 + 2*self.a2*k_param + 8*self.a4*(k_param**3) + 96*self.a6*(k_param**5)
        w_param = self.a1*k_param + self.a2*(k_param**2) - self.a2 + 3*self.a4*(k_param**4) - 6*self.a4*(k_param**2) - self.a4 + 35*self.a6*(k_param**6) - 75*self.a6*(k_param**4) - 15*self.a6*(k_param**2) - self.a6
        x0_param = self.x0_param
        frac = A1_param/(self.a_param*torch.exp(x-C0_param*t+x0_param) + self.khi/(4*self.a_param*torch.exp(x-C0_param*t+x0_param)))
        u = frac*torch.cos(k_param*x-w_param*t+self.theta0)
        v = frac*torch.sin(k_param*x-w_param*t+self.theta0)
        return u + 1j*v

class sixth_order_gaussian(sixth_order):
    #conventional gaussian function
    def q(self,x,t):
        return 1*np.exp(-(x-self.x0_param)**2) + 0*1j
    def q_tensor(self,x,t):
        return 1*torch.exp(-(x-self.x0_param)**2) + 0*1j