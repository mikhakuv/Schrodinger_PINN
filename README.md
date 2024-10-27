This tool is designed to solve one-dimensional Schrödinger equations using Physics-Informed Neural Networks technology, proposed by Raissi et al [1].

## Improvements
Although the basic approach remains same, many improvements are available:  
### Wise Points Generation  
Points can be generated with respect to residual value: higher residual corresponds to higher probability of being included into next training set. There are 3 different formulas of probability $P(x)$ which implement this idea:  

$$P_1(x) \propto \nu_1$$  

$$P_2(x) \propto \nu_1 + \lambda_1$$  

$$P_3(x) \propto \frac{\nu_1 + \nu_2}{2} + \lambda_2 \cdot max\left(\frac{\nu_1 + \nu_2}{2}\right)$$

Where:  

$$\nu_1 = \frac{res(x) - min(res)}{max(res) - min(res)},\quad \text{where}\quad res(x) - \text{residual of the output at}\ x$$  

$$\nu_2 = \frac{sum\\_abs(x) - min(sum\\_abs)}{max(sum\\_abs) - min(sum\\_abs)},\quad \text{where}\quad sum\\_abs(x)=|Re(q)| + |Im(q)|$$

$$\lambda_1\ \text{and}\ \lambda_2\ \text{are constants and automatically set up as}\ \lambda_1=0.005,\ \lambda_2 = 0.01$$  

To enable this option, set `PINN.points_gen_method` as `"first"`,`"second"` or `"third"` correspondingly. By default `PINN.points_gen_method = "random"`  
### Causal Loss  

### Loss Balancing  
Loss for PINN is defined as follows:  

$$Loss = \lambda_i\cdot Loss_{ic} + \lambda_b\cdot Loss_{bc} + \lambda_f\cdot Loss_{eq}$$  

Without active loss balancing, $\lambda_i,\ \lambda_b, \lambda_f$ are constant values, set as `PINN.lambda_i=10/12`, `PINN.lambda_b=1/12`, `PINN.lambda_f=1/12` by default. However, active loss balancing option is also available, namely ReloBRaLo method. It changes $\lambda_i,\ \lambda_b, \lambda_f$ dinamically according to the following formula:  

$$\lambda_n = \tau\cdot\left(\rho\cdot\lambda_n(iter-1) + (1-\rho)\cdot\widehat{\lambda_n}(iter)\right) + (1-\tau)\cdot\lambda_n(iter)$$  

$$\text{where}\ \lambda_n(iter) = \frac{exp\left(\frac{loss_n(iter)}{loss_n(iter-1)}\right)}{\sum\limits_{k} exp\left(\frac{loss_k(iter)}{loss_k(iter-1)}\right)},\ \widehat{\lambda_n}(iter) = \frac{exp\left(\frac{loss_n(iter)}{loss_n(0)}\right)}{\sum\limits_{k} exp\left(\frac{loss_k(iter)}{loss_k(0)}\right)},$$  

$$\tau \in[0,1] - \text{extinction coefficient},\ \rho \in[0,1] - \text{random lookback coefficient}$$

Set `PINN.loss_bal_method="relobralo"` for using active loss balancing method. By default `PINN.loss_bal_method="none"`.
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
