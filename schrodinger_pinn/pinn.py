import os, shutil
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from nys_newton_cg import NysNewtonCG
from collections import OrderedDict
from pyDOE import lhs

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class SinActivation(torch.nn.Module):
    def __init__(self):
        super(SinActivation, self).__init__()
        return
    def forward(self, x):
        return torch.sin(x)

class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()

        self.depth = len(layers) - 1
        self.activation = SinActivation #custom activation function

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1])))
            layer_list.append(('activation_%d' % i, self.activation()))
        layer_list.append(('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1])))
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)
    def forward(self, x):
        out = self.layers(x)
        return out

class PINN():
    def __init__(self, problem, layers, X_i, u, v, X_b, X_g):
        #verbosity
        self.verbosity = 100 #loss output frequency
        self.make_res_gif = False #makes gif with residual history
        #points generation options
        self.points_gen_method = "random" #"random"/first"/"second"/"third"
        self.points_gen_freq = 10 #points generation frequency
        self.points_am = 5000 #amount of collocation points
        #optimization options
        self.adam_steps = 1000
        self.lbfgs_steps = 10
        self.nncg_steps = 0
        self.adam_step_decay = 0.997
        self.lbfgs_step_decay = 0.990
        self.nncg_step_decay = 0.990
        self.decay_freq = 100
        #loss balancing options
        self.loss_bal_method = "none" #"none"/relobralo"
        self.bal_freq = 1 #loss rebalancing frequency
        self.lambda_i = 950/1000 #loss weights
        self.lambda_b = 49/1000
        self.lambda_f = 1/1000
        self.extinction = 0.9 #extinction coefficient for ReLoBRaLo
        #causal training
        self.causal_loss = False
        self.epsilon = 0.1
        self.t_partition = 30 #number of parts in the [t_0, t_1] division
        
        self.llc = np.array([problem.x_0, problem.t_0]) #left lower corner of domain
        self.ruc = np.array([problem.x_1, problem.t_1]) #right upper corner of domain
        self.x_i = torch.tensor(X_i[:, 0:1], requires_grad=True).float().to(device) #initial conditions: (x_i, t_i, u, v)
        self.t_i = torch.tensor(X_i[:, 1:2], requires_grad=True).float().to(device)
        self.u = torch.tensor(u).float().to(device)
        self.v = torch.tensor(v).float().to(device)
        self.x_b = torch.tensor(X_b[:, 0:1], requires_grad=True).float().to(device) #boundary conditions: (x_b, t_b, 0, 0)
        self.t_b = torch.tensor(X_b[:, 1:2], requires_grad=True).float().to(device)
        #if self.points_gen_method != "random" or self.make_res_gif: #grid of values for wise point generation or chart generation
        self.x_grid = torch.tensor(X_g[:,:,0].flatten()[:,np.newaxis], requires_grad=True).float().to(device)
        self.t_grid = torch.tensor(X_g[:,:,1].flatten()[:,np.newaxis], requires_grad=True).float().to(device)
        self.grid_shapes = X_g.shape
        self.x_f = torch.tensor((self.llc + (self.ruc-self.llc)*lhs(2, self.points_am))[:, 0:1], requires_grad=True).float().to(device) #initial collocation points
        self.t_f = torch.tensor((self.llc + (self.ruc-self.llc)*lhs(2, self.points_am))[:, 1:2], requires_grad=True).float().to(device)
        
        #if self.make_res_gif: #frames for gif creation
        self.frames=[]
        #self.snap_freq=(self.adam_steps+self.lbfgs_steps+self.nncg_steps)//100
        
        self.layers = layers
        self.dnn = DNN(layers).to(device)

        self.calc_res = problem.calc_res

        # optimizers
        self.adam = torch.optim.Adam(
          self.dnn.parameters(),
          lr=0.01,
          betas=(0.9, 0.999),
          eps=1e-08,
          weight_decay=0,
          amsgrad=False)
        
        self.lbfgs = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe")
        
        self.nncg = NysNewtonCG(
            self.dnn.parameters(),
            lr=0.005,
            rank=10,
            mu=1e-4,
            cg_tol=1e-5,
            cg_max_iters=100,
            line_search_fn='armijo')

        self.initial_lambda_i = self.lambda_i
        self.initial_lambda_b = self.lambda_b
        self.initial_lambda_f = self.lambda_f
        self.nncg_active = False
        self.first_nncg_iter = False
        self.grad_tuple = None
        self.train_finished = False
        self.iter = 0
        self.loss_hist = []
        self.iter_hist = []
        
        #if self.loss_bal_method == "relobralo": # initial values for relobralo method
        u_init, v_init = self.net_uv(self.x_i, self.t_i)
        u_bound, v_bound = self.net_uv(self.x_b, self.t_b)
        f_u_pred, f_v_pred = self.net_f(self.x_f, self.t_f)
        #initial condition:
        self.L_in = torch.mean(((self.u - u_init)**2 + (self.v - v_init)**2)/2).item()
        #boundary conditions:
        u_x = torch.autograd.grad(u_bound,self.x_b,grad_outputs=torch.ones_like(u_bound),retain_graph=True,create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x,self.x_b,grad_outputs=torch.ones_like(u_x),retain_graph=True,create_graph=True)[0]
        v_x = torch.autograd.grad(v_bound,self.x_b,grad_outputs=torch.ones_like(v_bound),retain_graph=True,create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x,self.x_b,grad_outputs=torch.ones_like(v_x),retain_graph=True,create_graph=True)[0]
        self.L_b = torch.mean((u_bound**2 + u_x**2 + u_xx**2 + v_bound**2 + v_x**2 + v_xx**2)/6).item()
        #equation condition:
        self.L_f = torch.mean((f_u_pred**2 + f_v_pred**2)/2).item()
        self.L_in_0 = self.L_in
        self.L_b_0 = self.L_b
        self.L_f_0 = self.L_f

    def net_uv(self, x, t): #model output
        u = self.dnn(torch.cat([x, t], dim=1))[:,0:1]
        v = self.dnn(torch.cat([x, t], dim=1))[:,1:2]
        return u, v

    def net_f(self, x, t): #equation output
        u, v = self.net_uv(x,t)
        f_u, f_v = self.calc_res(u,v,x,t)
        return f_u, f_v

    def loss_func(self):
        self.adam.zero_grad()
        self.lbfgs.zero_grad()
        self.nncg.zero_grad()

        u_init, v_init = self.net_uv(self.x_i, self.t_i)
        u_bound, v_bound = self.net_uv(self.x_b, self.t_b)
        f_u_pred, f_v_pred = self.net_f(self.x_f, self.t_f)
        #initial condition:
        loss_i = torch.mean(((self.u - u_init)**2 + (self.v - v_init)**2)/2)
        #boundary conditions:
        u_x = torch.autograd.grad(u_bound,self.x_b,grad_outputs=torch.ones_like(u_bound),retain_graph=True,create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x,self.x_b,grad_outputs=torch.ones_like(u_x),retain_graph=True,create_graph=True)[0]
        v_x = torch.autograd.grad(v_bound,self.x_b,grad_outputs=torch.ones_like(v_bound),retain_graph=True,create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x,self.x_b,grad_outputs=torch.ones_like(v_x),retain_graph=True,create_graph=True)[0]
        loss_b = torch.mean((u_bound**2 + u_x**2 + u_xx**2 + v_bound**2 + v_x**2 + v_xx**2)/6)
        #equation condition:
        loss_f = torch.mean((f_u_pred**2 + f_v_pred**2)/2)
        if self.causal_loss == True and self.train_finished == False and loss_f<1: #last condition implies protection against loss_f spikes
            residuals = (f_u_pred**2 + f_v_pred**2)/2
            t_interval = (self.ruc[1] - self.llc[1])/self.t_partition
            loss_f=0
            sum_residuals=0
            sum_w=0
            for i in range(self.t_partition): #calculating loss_f=sum(w_i*loss_f_i) according to causal training
                t_l=self.llc[1]+i*t_interval
                t_r=t_l+t_interval
                current_residuals=residuals[torch.where((self.t_f>=t_l) & (self.t_f<t_r))]
                sum_residuals+=torch.mean(current_residuals).item()
                current_w=math.exp(-self.epsilon*sum_residuals)
                sum_w+=current_w
                loss_f+=torch.mean(current_residuals)*current_w
            loss_f=loss_f/sum_w
            
        #loss evaluation for trained PINN
        if self.train_finished == True:
            loss = self.lambda_i*loss_i + self.lambda_b*loss_b + self.lambda_f*loss_f
            return loss
        
        if self.loss_bal_method == "relobralo" and self.iter%self.bal_freq == 0:
            L_in_prev = self.L_in
            L_b_prev = self.L_b
            L_f_prev = self.L_f
            self.L_in = loss_i.item()
            self.L_b = loss_b.item()
            self.L_f = loss_f.item()
            random_lookback = np.random.random()
            if self.L_f/self.L_f_0<1 and self.L_f/L_f_prev<1: #protection against loss_f spikes
                lambda_i_0 = math.exp(self.L_in/self.L_in_0)/(math.exp(self.L_in/self.L_in_0) + math.exp(self.L_b/self.L_b_0) + math.exp(self.L_f/self.L_f_0))
                lambda_b_0 = math.exp(self.L_b/self.L_b_0)/(math.exp(self.L_in/self.L_in_0) + math.exp(self.L_b/self.L_b_0) + math.exp(self.L_f/self.L_f_0))
                lambda_f_0 = math.exp(self.L_f/self.L_f_0)/(math.exp(self.L_in/self.L_in_0) + math.exp(self.L_b/self.L_b_0) + math.exp(self.L_f/self.L_f_0))
                lambda_i_prev = math.exp(self.L_in/L_in_prev)/(math.exp(self.L_in/L_in_prev) + math.exp(self.L_b/L_b_prev) + math.exp(self.L_f/L_f_prev))
                lambda_b_prev = math.exp(self.L_b/L_b_prev)/(math.exp(self.L_in/L_in_prev) + math.exp(self.L_b/L_b_prev) + math.exp(self.L_f/L_f_prev))
                lambda_f_prev = math.exp(self.L_f/L_f_prev)/(math.exp(self.L_in/L_in_prev) + math.exp(self.L_b/L_b_prev) + math.exp(self.L_f/L_f_prev))
                self.lambda_i = self.extinction*(random_lookback*self.lambda_i + (1-random_lookback)*lambda_i_0) + (1-self.extinction)*lambda_i_prev
                self.lambda_b = self.extinction*(random_lookback*self.lambda_b + (1-random_lookback)*lambda_b_0) + (1-self.extinction)*lambda_b_prev
                self.lambda_f = self.extinction*(random_lookback*self.lambda_f + (1-random_lookback)*lambda_f_0) + (1-self.extinction)*lambda_f_prev
        
        loss = self.lambda_i*loss_i + self.lambda_b*loss_b + self.lambda_f*loss_f
        
        self.iter += 1
        self.loss_hist.append(loss.item())
        self.iter_hist.append(self.iter)

        if self.iter % self.verbosity == 0:
            print('Iter %d, Loss: %.3e, Loss_i: %.2e, Loss_b: %.2e, Loss_f: %.2e' % (self.iter, loss.item(), loss_i.item(), loss_b.item(), loss_f.item()))
        if self.make_res_gif and self.iter%self.snap_freq==0:
            f_u_pred, f_v_pred = self.net_f(self.x_grid, self.t_grid)
            residual = ((f_u_pred**2 + f_v_pred**2)**0.5).detach().cpu().numpy().reshape((self.grid_shapes[0],self.grid_shapes[1]))
            x_grid=self.x_grid.detach().cpu().numpy().reshape((self.grid_shapes[0],self.grid_shapes[1]))
            t_grid=self.t_grid.detach().cpu().numpy().reshape((self.grid_shapes[0],self.grid_shapes[1]))
            fig, ax = plt.subplots()
            c = ax.pcolormesh(t_grid, x_grid, residual, shading="nearest", cmap='Reds', vmin=0, vmax=0.1)
            ax.set_title(f'iter={self.iter}')
            ax.axis([np.min(t_grid), np.max(t_grid), np.min(x_grid), np.max(x_grid)])
            fig.colorbar(c, ax=ax)
            plt.savefig(f'frames/capture_{self.iter}.png')
            plt.close()
            frame = Image.open(f'frames/capture_{self.iter}.png')
            self.frames.append(frame)

        if self.iter % self.points_gen_freq == 0: #generating new points
            lambda_1 = 0.005
            lambda_2 = 0.01
            if self.points_gen_method == "random": #random points choice
                random_points = self.llc + (self.ruc-self.llc)*lhs(2, self.points_am)
                self.x_f = torch.tensor(random_points[:, 0:1], requires_grad=True).float().to(device)
                self.t_f = torch.tensor(random_points[:, 1:2], requires_grad=True).float().to(device)
            else:
                f_u_pred, f_v_pred = self.net_f(self.x_grid, self.t_grid)
                residual = ((f_u_pred**2 + f_v_pred**2)**0.5).detach().cpu().numpy()
                max_residual = np.max(residual)
                min_residual = np.min(residual)
                nu_1 = (residual - min_residual)/(max_residual - min_residual)
                if self.points_gen_method == "first": probability = nu_1
                if self.points_gen_method == "second": probability = nu_1 + lambda_1
                if self.points_gen_method == "third":
                    u_pred, v_pred = self.net_uv(self.x_grid, self.t_grid)
                    sum_abs = (torch.abs(u_pred) + torch.abs(v_pred)).detach().cpu().numpy()
                    max_sum_abs = np.max(sum_abs)
                    min_sum_abs = np.min(sum_abs)
                    nu_2 = (sum_abs - min_sum_abs)/(max_sum_abs - min_sum_abs)
                    nu_3 = (nu_1 + nu_2)/2
                    probability = nu_3 + lambda_2*np.max(nu_3)
                norm_probability = probability[:,0]/np.sum(probability)
                inds = np.random.choice(self.grid_shapes[0]*self.grid_shapes[1], self.points_am, replace=False, p=norm_probability)
                self.x_f = torch.clone(self.x_grid[inds,:])
                self.t_f = torch.clone(self.t_grid[inds,:])

        if self.nncg_active == True:
            self.grad_tuple = torch.autograd.grad(loss, self.dnn.parameters(), create_graph=True)
            if self.first_nncg_iter == True:
                self.nncg.update_preconditioner(self.grad_tuple)
                self.first_nncg_iter = False
            return loss, self.grad_tuple
        else:
            loss.backward(retain_graph=True)
            return loss

    def train(self):        
        if self.make_res_gif:
            self.snap_freq=(self.adam_steps+self.lbfgs_steps+self.nncg_steps)//100
            if not os.path.exists('./frames'):
                os.mkdir('./frames')
        lbfgs_iterations=0
        nncg_iterations=0
        self.train_finished = False
        if self.iter == 0:
            if self.causal_loss == True:
                print(f'Training started with {self.points_gen_method} points generation method, {self.loss_bal_method} loss balancing and causal loss')
            else:
                print(f'Training started with {self.points_gen_method} points generation method and {self.loss_bal_method} loss balancing')
        if self.iter > 0:
            if self.causal_loss == True:
                print(f'Training continued from Iter={self.iter} with {self.points_gen_method} points generation method, {self.loss_bal_method} loss balancing and causal loss')
            else:
                print(f'Training continued from Iter={self.iter} with {self.points_gen_method} points generation method and {self.loss_bal_method} loss balancing')
        
        print(f'{self.adam_steps} steps of ADAM:')
        self.dnn.train()
        for i in range(self.adam_steps):
            self.adam.step(self.loss_func)
            if i % self.decay_freq == 0: self.adam.param_groups[0]['lr'] = self.adam_step_decay*self.adam.param_groups[0]['lr']
        if self.lbfgs_steps != 0:
            print(f'{self.lbfgs_steps} steps of LBFGS:')
            for i in range(self.lbfgs_steps):
                self.lbfgs.step(self.loss_func)
                if i % self.decay_freq == 0: self.lbfgs.param_groups[0]['lr'] = self.lbfgs_step_decay*self.lbfgs.param_groups[0]['lr']
            lbfgs_iterations = self.iter-self.adam_steps
        if self.nncg_steps != 0:
            print(f'{self.nncg_steps} steps of NNCG:')
            self.nncg_active = True
            for i in range(self.nncg_steps):
                self.first_nncg_iter = True
                self.nncg.step(self.loss_func)
                if i % self.decay_freq == 0: self.nncg.param_groups[0]['lr'] = self.nncg_step_decay*self.nncg.param_groups[0]['lr']
            nncg_iterations = self.iter-lbfgs_iterations-self.adam_steps
        print(f'Total iterations: {self.adam_steps} + {lbfgs_iterations} + {nncg_iterations}')
        if self.make_res_gif:
            self.frames[0].save('train_process.gif', save_all=True, append_images=self.frames[1:],optimize=True,duration=100,loop=0)
            self.frames=[]
            shutil.rmtree('./frames')
        self.train_finished = True
        self.lambda_i = self.initial_lambda_i
        self.lambda_b = self.initial_lambda_b
        self.lambda_f = self.initial_lambda_f

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        self.dnn.eval()
        u, v = self.net_uv(x, t)
        f_u, f_v = self.net_f(x, t)
        u = u.detach().cpu().numpy()
        v = v.detach().cpu().numpy()
        f_u = f_u.detach().cpu().numpy()
        f_v = f_v.detach().cpu().numpy()
        return u, v, f_u, f_v
    
    def train_hist(self, logscale=True, step=1):
        plt.plot(np.array(self.iter_hist)[::step], np.array(self.loss_hist)[::step], color = "blue")
        plt.figtext(0.15, -0.05, f'Final loss: {self.loss_func()}', weight='regular', fontsize='9')
        #plt.ylim(0,5e-3)
        if logscale:
            plt.yscale("log")
        plt.title('Loss(iter)')
        return plt.show()
    
    def plot_residual(self, X, T):
        X_tensor = torch.tensor(X.flatten()[:,np.newaxis], requires_grad=True).float()
        T_tensor = torch.tensor(T.flatten()[:,np.newaxis], requires_grad=True).float()
        Real_Loss, Complex_Loss = self.net_f(X_tensor.to(device), T_tensor.to(device))
        flatten_Loss = (Real_Loss**2 + Complex_Loss**2)**0.5
        Loss = torch.reshape(flatten_Loss,(X.shape[0],X.shape[1])).cpu().detach().numpy()
        fig, ax = plt.subplots(1, 1, figsize=(6,5), dpi=100)
        plt.title('Residual(t,x)')
        ax.set(xlabel='$t$', ylabel='$x$')
        c = ax.pcolormesh(T, X, Loss, shading='nearest', cmap='Reds')
        ax.axis([T.min(), T.max(), X.min(), X.max()])
        fig.colorbar(c, ax=ax)
        return plt.show()
        
    def clear(self): #clears all weights and history, but retains hyperparameters (this method is used in params tuning)
        self.dnn = DNN(self.layers).to(device)
        self.nncg_active = False
        self.first_nncg_iter = False
        self.grad_tuple = None
        self.train_finished = False
        self.iter = 0
        self.loss_hist = []
        self.iter_hist = []
        if self.make_res_gif:
            self.frames=[]
        if self.loss_bal_method == "relobralo":
            u_init, v_init = self.net_uv(self.x_i, self.t_i)
            u_bound, v_bound = self.net_uv(self.x_b, self.t_b)
            f_u_pred, f_v_pred = self.net_f(self.x_f, self.t_f)
            #initial condition:
            self.L_in = torch.mean(((self.u - u_init)**2 + (self.v - v_init)**2)/2).item()
            #boundary conditions:
            u_x = torch.autograd.grad(u_bound,self.x_b,grad_outputs=torch.ones_like(u_bound),retain_graph=True,create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x,self.x_b,grad_outputs=torch.ones_like(u_x),retain_graph=True,create_graph=True)[0]
            v_x = torch.autograd.grad(v_bound,self.x_b,grad_outputs=torch.ones_like(v_bound),retain_graph=True,create_graph=True)[0]
            v_xx = torch.autograd.grad(v_x,self.x_b,grad_outputs=torch.ones_like(v_x),retain_graph=True,create_graph=True)[0]
            self.L_b = torch.mean((u_bound**2 + u_x**2 + u_xx**2 + v_bound**2 + v_x**2 + v_xx**2)/6).item()
            #equation condition:
            self.L_f = torch.mean((f_u_pred**2 + f_v_pred**2)/2).item()
            self.L_in_0 = self.L_in
            self.L_b_0 = self.L_b
            self.L_f_0 = self.L_f
        self.adam = torch.optim.Adam(self.dnn.parameters(),lr=0.01,betas=(0.9, 0.999),eps=1e-08,weight_decay=0,amsgrad=False)
        self.lbfgs = torch.optim.LBFGS(self.dnn.parameters(),lr=1.0,max_iter=50000,max_eval=50000,history_size=50,tolerance_grad=1e-5,tolerance_change=1.0 * np.finfo(float).eps,line_search_fn="strong_wolfe")
        self.nncg = NysNewtonCG(self.dnn.parameters(),lr=0.005,rank=10,mu=1e-4,cg_tol=1e-5,cg_max_iters=100,line_search_fn='armijo')
        return 0

def make_points(problem, init_points_amt, bound_points_amt, grid_resolution_x, grid_resolution_t):
    x_0, x_1 = problem.x_0, problem.x_1
    t_0, t_1 = problem.t_0, problem.t_1
    x = np.linspace(x_0, x_1, init_points_amt)
    t = np.linspace(t_0, t_1, bound_points_amt)
    X, T = np.meshgrid(x, t)
    # initial condition
    xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T)) #(x,t_0)
    uu1 = np.real(problem.q(X[0:1,:].T, T[0:1,:].T))
    vv1 = np.imag(problem.q(X[0:1,:].T, T[0:1,:].T))
    XT_init = xx1
    U_init = uu1
    V_init = vv1
    # boundary conditions
    xx2 = np.hstack((X[:,0:1], T[:,0:1])) #(x_0,t)
    xx3 = np.hstack((X[:,-1:], T[:,-1:])) #(x_1,t)
    #xx4 = np.hstack((X[-1:,:].T, T[-1:,:].T)) #(x,t_1)
    XT_bound = np.vstack([xx2, xx3])
    #grid for wise points generation
    X_g, T_g = np.meshgrid(np.linspace(x_0, x_1, grid_resolution_x), np.linspace(t_0, t_1, grid_resolution_t))
    XT_grid = np.stack((X_g, T_g),axis=2)
    return XT_init, U_init, V_init, XT_bound, XT_grid
