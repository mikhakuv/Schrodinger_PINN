This tool is designed to solve one-dimensional Schrödinger equations using Physics-Informed Neural Networks technology, proposed by Raissi et al [1].

## Improvements
Although the basic approach remains same, many improvements are available:  
### Wise Points Generation  
Points can be generated with respect to residual value: higher residual corresponds to higher probability of being included into next training set. There are 3 different formulas of probability $P(x)$ which implement this idea:  

$$P_1(x) \propto \frac{res(x) - min(res)}{max(res) - min(res)},\ \text{where}\ res(x) - \text{residual of the output at}\ x$$  

$$P_2(x) \propto P_1(x) + \lambda_1$$  

$$P_3(x) \propto 0.5\cdot(P_1(x)+\frac{sum\\_abs(x) - min(sum\\_abs)}{max(sum\\_abs) - min(sum\\_abs)}) + \lambda_2 \cdot max(),\ \text{where}\ sum\\_abs(x)=|Re(q(x))| + |Im(q(x))|$$

$\lambda_1$ and $\lambda_2$ are constants and automatically set up as $\lambda_1=0.005$, $\lambda_2 = 0.01$.  
To enable this option, set `PINN.points_gen_method` as `"first"`,`"second"` or `"third"` correspondingly. By default `PINN.points_gen_method = "random"`  
### Causal Loss  

### Loss Balancing  
Active loss balancing is disabled by default, yet optionally ReLoBRaLo method is available as option.  

$$lambda_i = \tau*(\rho*lambda_i(iter-1) + (1-\rho) *\widehat{lambda_i}(iter)) + (1-\tau)*lambda_i(iter)$$  

где $\tau \in[0,1]$ - коэффициент, обозначающий затухание, $\rho \in[0,1]$ - случайное число, генерируется каждую итерацию,  

$$\widehat{\lambda_i} = \frac{exp(\frac{loss_i(iter)}{loss_i(0)})}{\sum\limits_{j} exp(\frac{loss_j(iter)}{loss_j(0)})}$$  

### Optimizers  
Three optimizers are available: ADAM, LBFGS and NNCG. Amount of steps by each can be set using `PINN.adam_steps`, `PINN.lbfgs_steps` and `PINN.nncg_steps`. Moreover, exponential step decay can be established by `PINN.adam_step_decay`, `PINN.lbfgs_step_decay` `PINN.nncg_step_decay` combined with `PINN.decay_freq` which defines frequency of decay.  

## Plots and Statistics
For accuracy evaluation plenty metrics and visualizations are available:  


## Problems  
Two problems are given as an example:
### 2nd order  
### 6th order  

Analytical solutions for considered nonlinear Schrodinger equations are described in [] - second order and [] - third order.
## Literature
**1** *M. Raissi, P. Perdikaris, G.E. Karniadakis* Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, Journal of Computational Physics, Volume 378, 2019, Pages 686-707  
**2**   
**3**  
