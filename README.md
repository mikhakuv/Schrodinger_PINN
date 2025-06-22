This tool is designed to solve one-dimensional Schrödinger equation using Physics-Informed Neural Networks technology, proposed by Raissi et al [[1]](https://github.com/mikhakuv/Schrodinger_PINN?tab=readme-ov-file#literature).  

<!--
> UPDATE: New tools are implemented. It is now possible to utilize SP_PINN, a network that features a new architecture. SP_PINN represents separate neural networks for each of the coordinates, and its output is calculated as a scalar product of the outputs of individual networks (the idea is very similar to the Separable PINN, proposed at [[9]](https://github.com/mikhakuv/Schrodinger_PINN?tab=readme-ov-file#literature)). Additionally, Seg_PINN is also implemented. This segmentation model divides the region into segments and sequentially trains a different PINN on each segment, the resulting solutions are then merged using one global PINN. Examples are available for further study: [SP_PINN](https://github.com/mikhakuv/Schrodinger_PINN/tree/main/examples/SP_PINN.ipynb), [Seg_PINN](https://github.com/mikhakuv/Schrodinger_PINN/tree/main/examples/Seg_PINN.ipynb).
-->

## Installation
We recommend to create new Python environment for using `Schrodinger PINN`:  
```bash
conda create -p $HOME/environments/schrod_pinn python=3.9
conda activate $HOME/environments/schrod_pinn
git clone git@github.com:mikhakuv/Schrodinger_PINN.git
pip install -e .
python -m ipykernel install --user --name schrod_pinn --display-name "schrod_pinn kernel"
```
Next, choose kernel "schrod_pinn kernel" before running the notebooks. Illustrational usage examples can be found at [the folder](https://github.com/mikhakuv/Schrodinger_PINN/tree/main/examples).

## Getting started
1. Create custom problem class or choose one from existing, e.g.:
   ```python
    from problems import sixth_order
    #domain setting
    x_0=-10.
    x_1=10.
    t_0=0.
    t_1=1.
    #problem setting
    a1 = 1
    a2 = -1
    a4 = -0.3
    a6 = 0.1
    b1 = 6
    khi = 1
    a_param = 0.5
    x0_param = 4
    theta0 = math.pi/3
    problem = sixth_order(x_0, x_1, t_0, t_1, a1, a2, a4, a6, b1, khi, a_param, x0_param, theta0)
   ```
   For description of defined problems, see [Problems](https://github.com/mikhakuv/Schrodinger_PINN/tree/main?tab=readme-ov-file#problems).
3. Generate set of training data using `make_points`:
   ```python
   X_i, u, v, X_b, X_g = make_points(problem, init_points_amt=400, bound_points_amt=100, grid_resolution_x=200, grid_resolution_t=100)
   ```
   * **problem** - problem under consideration
   * **init_points_amt** - amount of points on initial condition
   * **bound_points_amt** - resolution of points on boundary conditions
   * **grid_resolution_x** and **grid_resolution_t** - $x$ and $t$ resolutions of grid used for wise points generation
4. Define model:
   ```python
   model = PINN(problem, layers=[2,100,100,100,2], X_i, u, v, X_b, X_g)
   ```
   * **problem** - problem class element as defined earlier
   * **layers** - topology of the network. Since total amount of input dimensions is 2 ($x$, $t$) as well as total amount of output dimensions (real and imaginary parts), first and last numbers must be 2
   * **X_i**, **u**, **v**, **X_b** and **X_g** are train points and values generated before by using `make_points` function
6. Configure the training process, simplest settings are listed bellow:  
   ```python
   model.verbosity = 1000 #frequency of loss output
   model.points_am = 5000  #amount of collocation points
   model.adam_steps = 10000  #amount of steps by primordial optimizator
   ```
   Additionally, there are many improvements available, for more information see [Improvements](https://github.com/mikhakuv/Schrodinger_PINN/tree/main?tab=readme-ov-file#improvements).
7. Train:
   ```python
   model.train()
   ```
8. Evaluate performance, e.g.:
   ```python
   x=np.linspace(x_0,x_1,200)
   t=np.linspace(t_0,t_1,100)
   X, T = np.meshgrid(x, t)
   model.plot_residual(X, T)
   ```
    There are many other charts and statistics available, see [Plots and Statistics](https://github.com/mikhakuv/Schrodinger_PINN/tree/main?tab=readme-ov-file#plots-and-statistics).

## Improvements
Although the basic approach remains same as described in [[1]](https://github.com/mikhakuv/Schrodinger_PINN?tab=readme-ov-file#literature), many improvements are available:  
### Wise Points Generation  
Points can be generated with respect to residual value: higher residual corresponds to higher probability of being included into next training set. There are 3 different formulas of probability $P(x)$ which implement this idea:  

$$P_1(x) \propto \nu_1$$  

$$P_2(x) \propto \nu_1 + \lambda_1$$  

$$P_3(x) \propto \frac{\nu_1 + \nu_2}{2} + \lambda_2 \cdot max\left(\frac{\nu_1 + \nu_2}{2}\right)$$

Where:  

$$\nu_1 = \frac{res(x) - min(res)}{max(res) - min(res)},\quad \text{where}\quad res(x) - \text{residual of the output at}\ x$$  

$$\nu_2 = \frac{sum\\_abs(x) - min(sum\\_abs)}{max(sum\\_abs) - min(sum\\_abs)},\quad \text{where}\quad sum\\_abs(x)=|Re(q)| + |Im(q)|$$

$$\lambda_1\ \text{and}\ \lambda_2\ \text{are constants and automatically set up as}\ \lambda_1=0.005,\ \lambda_2 = 0.01$$  

The idea was proposed in [[2]](https://github.com/mikhakuv/Schrodinger_PINN?tab=readme-ov-file#literature). To enable this option, set `PINN.points_gen_method` as `"first"`,`"second"` or `"third"` correspondingly. By default `PINN.points_gen_method = "random"`  
### Causal Loss  
By default, loss on equation is calculated as follows:  

$$Loss_{eq} = \frac{1}{N_t}\sum_{k=1}^{N_t} Loss_{eq}(t_k),\ \text{where}\ Loss_{eq}(t_k)\ \text{is mean residual on points with}\ t=t_k$$  

Causal loss employs differrent formula which represents the idea of causality in training suggested in [[3]](https://github.com/mikhakuv/Schrodinger_PINN?tab=readme-ov-file#literature):  

$$Loss_{eq} = \frac{1}{N_t\cdot\sum_{k=1}^{N_t}w_k}\sum_{k=1}^{N_t} Loss_{eq}(t_k)\cdot w_k$$  

$$\text{where}\ w_k = exp\left(-\varepsilon\cdot\sum_{i=1}^{k-1}Loss_{eq}(t_k)\right)\ \text{and}\ \varepsilon\ \text{is changeable parameter}$$  

Causality means that if $Loss_{eq}(t)$ is high for $t<\tilde{t}$, then PINN will not be training on $t>\tilde{t}$ until $Loss_{eq}(t)$ for $t<\tilde{t}$ gets lower. To enable this option, set `PINN.causal_loss = True` and adjust `PINN.epsilon`, `PINN.t_partition`.  

### Loss Balancing  
Loss for PINN is defined as follows:  

$$Loss = \lambda_i\cdot Loss_{ic} + \lambda_b\cdot Loss_{bc} + \lambda_f\cdot Loss_{eq}$$  

Without active loss balancing, $\lambda_i,\ \lambda_b, \lambda_f$ are constant values, set as `PINN.lambda_i=950/1000`, `PINN.lambda_b=49/1000`, `PINN.lambda_f=1/1000` by default. However, active loss balancing option is also available, namely ReloBRaLo method which is introduced in [[4]](https://github.com/mikhakuv/Schrodinger_PINN?tab=readme-ov-file#literature). It changes $\lambda_i,\ \lambda_b, \lambda_f$ dinamically according to the following formula:  

$$\lambda_n = \tau\cdot\left(\rho\cdot\lambda_n(iter-1) + (1-\rho)\cdot\widehat{\lambda_n}(iter)\right) + (1-\tau)\cdot\lambda_n(iter)$$  

$$\text{where}\ \lambda_n(iter) = \frac{exp\left(\frac{loss_n(iter)}{loss_n(iter-1)}\right)}{\sum\limits_{k} exp\left(\frac{loss_k(iter)}{loss_k(iter-1)}\right)},\ \widehat{\lambda_n}(iter) = \frac{exp\left(\frac{loss_n(iter)}{loss_n(0)}\right)}{\sum\limits_{k} exp\left(\frac{loss_k(iter)}{loss_k(0)}\right)},$$  

$$\tau \in[0,1] - \text{extinction coefficient},\ \rho \in[0,1] - \text{random lookback coefficient}$$

Set `PINN.loss_bal_method="relobralo"` for using active loss balancing method. By default `PINN.loss_bal_method="none"`.
### Optimizers  
Three optimizers are available: ADAM, LBFGS and NNCG (see [[5]](https://github.com/mikhakuv/Schrodinger_PINN?tab=readme-ov-file#literature) for details). Amount of steps by each can be set using `PINN.adam_steps`, `PINN.lbfgs_steps` and `PINN.nncg_steps`. Moreover, exponential step decay can be established by `PINN.adam_step_decay`, `PINN.lbfgs_step_decay` `PINN.nncg_step_decay` combined with `PINN.decay_freq` which defines frequency of decay.  

### Segmentation
One possible solution for large-scale problems is to employ an intuitive "divide-and-conquer" approach, which entails dividing a large area into several segments, each with its own optimization problem. Separate neural networks can then be trained to solve each sub-problem independently. In this repository, this idea is implemented for the case shown in the scheme below:

<img src="https://github.com/mikhakuv/Schrodinger_PINN/blob/main/pictures/domain_decomposition.png">  

This segmentation model divides the region into segments and sequentially trains a different PINN on each segment, the resulting solutions are then merged using one global PINN. Usage example is available for further study: [Seg_PINN](https://github.com/mikhakuv/Schrodinger_PINN/tree/main/examples/Seg_PINN.ipynb).

## Plots and Statistics
### Metrics
For accuracy evaluation, following metrics are used:  

$$Rel_h= \frac{\sqrt{\sum_{n=1}^{N} (|\hat{q_{n}}| - |q_{n}|)^2}}{\sqrt{\sum_{i=1}^{N} (q_{i})^2}}$$  

$$max(Lw_1) = \max_{i \in T} \left ( \frac{|I_1(t_0) - \hat{I}_1(t_i)|}{I_1(t_0)} \right ) \cdot 100\\%$$  

$$mean(Lw_1) = \text{mean}_{i \in T} \left ( \frac{|I_1(t_0) - \hat{I}_1(t_i)|}{I_1(t_0)} \right ) \cdot 100\\%$$  

$$max(Lw_2) = \max_{i \in T} \left (\frac{|I_2(t_0) - \hat{I}_2(t_i)|}{I_2(t_0)} \right ) \cdot 100\\%$$  

$$mean(Lw_2) = \text{mean}_{i \in T} \left (\frac{|I_2(t_0) - \hat{I}_2(t_i)|}{I_2(t_0)} \right ) \cdot 100\\%$$  

$$\text{where}\ I_1(t) = \int_{-\infty}^{+\infty} |q(x,t)|^2 dx;\quad I_2(t) = \int_{-\infty}^{+\infty} (q^{\ast} q_x - q^{\ast}_x q) dx$$

$$\text{and hatted values are predicted values while unhatted values are ground truth}$$  

Plenty visualizing options are available:  
### Training History
`PINN.train_hist(logscale=bool, step=int)` returns loss history with `step` intervals using or not using logscale:  

<img src="https://github.com/mikhakuv/Schrodinger_PINN/blob/main/pictures/train_hist.png">  

### Residual of Solution
`PINN.plot_residual(X=np.ndarray, T=np.ndarray)` plots residual of obtained solution:  

<img src="https://github.com/mikhakuv/Schrodinger_PINN/blob/main/pictures/plot_residual.png">  

### Other Plots
`plot_comparison(X=np.ndarray, T=np.ndarray, Q_pred=np.ndarray, Q_truth=np.ndarray, savefig=bool, namefig=string)` shows comparison of `Q_pred` and `Q_truth`:  

<img src="https://github.com/mikhakuv/Schrodinger_PINN/blob/main/pictures/plot_comparison.png">  

`plot_errors(X=np.ndarray, T=np.ndarray, Q_pred=np.ndarray, Q_truth=np.ndarray, savefig=bool, namefig=string, savetable=bool, nametable=string)` uses `Q_pred` and `Q_truth` to plot $|q(x,t)|$, $Lw_1(t)$, $Lw_2(t)$ and $Rel_h(t)$ and return dictionary with evaluated metrics:  

<img src="https://github.com/mikhakuv/Schrodinger_PINN/blob/main/pictures/plot_errors.png">  

## Problems  
Three problems with reference analytical solution are given as example:
### 2nd order  
equation:  

$$iq_t + q_{xx} + q (|q|^2 + \alpha |q|^4 + \beta |q|^6) = 0$$  

specific solution in case $\beta = 0$ is given in [[6]](https://github.com/mikhakuv/Schrodinger_PINN?tab=readme-ov-file#literature):  

$$q(x,t)=\sqrt{\frac{4 \mu e^{\sqrt{\mu}(x-2k t - z_0)}}{1 + 4 e^{\sqrt{\mu}(x-2k t - z_0)} + (4 + 4 \mu \nu) e^{2\sqrt{\mu}(x-2k t - z_0)}}}\cdot e^{i(kx-wt+\theta_0)}$$  

$$\text{where}\ k,\ w,\ x_0,\ \theta_0\ \text{changeable, while other parameters are found using following expressions:}$$  

$$\mu = 4(k^2 - w),\quad \nu = \frac{4}{3}\alpha$$  

### 4th order  
equation:  

$$iq_t + ia_1q_x + a_2q_{xx} + ia_3q_{3x} + a_4q_{4x} - q (b_1|q|^2 + b_2|q|^4 + b_3|q|^6 + b_4|q|^8) = 0 $$   

specific solution is given in [[7]](https://github.com/mikhakuv/Schrodinger_PINN?tab=readme-ov-file#literature):  

$$q(x,t)=\frac{\sqrt{2A\nu}}{\sqrt{-\mu + \sqrt{\mu^2 - 4\nu}\cosh{(2\sqrt{\nu}(x-vt-\tilde{x}_0))}}}\cdot e^{i(kx-wt+\theta_0)}$$  

$$\text{where}\ a_1,\ a_2,\ a_4,\ A,\ k,\ \nu,\ \mu,\ \tilde{x}_0,\ \theta_0\ \text{changeable, while other parameters are found using following expressions:}$$  

$$v = 8k^3a_4 + 2ka_2 + a_1$$  

$$w = (3k^4 - 6\nu k^2 - \nu^2)a_4 + (k^2-\nu)a_2 + ka_1$$  

$$a_3 = -4ka_4$$  

$$b_1 = \frac{(12\mu k^2 + 20\mu\nu)a_4+2\mu a_2}{A}$$  

$$b_2 = \frac{(18k^2 + 24\mu^2 + 78\nu)a_4+3 a_2}{A^2}$$  

$$b_3 = \frac{120\mu a_4}{A^3},\quad b_4 = \frac{105a_4}{A^4}$$  

### 6th order  
equation:  

$$iq_t + ia_1q_x + a_2q_{xx} + ia_3q_{3x} + a_4q_{4x} + ia_5q_{5x} + a_6q_{6x} + q(b_1|q|^2 +b_2|q|^4 + b_3|q|^6)=0$$  

specific solution is found in [[7]](https://github.com/mikhakuv/Schrodinger_PINN?tab=readme-ov-file#literature):  

$$q(x,t) = \frac{A_1}{a e^{(x-C_0 t + x_0)} + \frac{\chi}{4a}\cdot e^{-(x-C_0 t + x_0)}}\cdot e^{i(kx-wt+\theta_0)}$$

$$\text{where}\ a_1,\ a_2,\ a_4,\ a_6,\ b_1,\ \chi,\ a,\ x_0,\ \theta_0 - \text{changeable, while other parameters are found using following expressions:}$$

$$k=1$$  

$$A_1 = \sqrt{2\chi\cdot\frac{a_2-6a_4 k^2 + 12a_4 k + 10a_4 + 75a_6 k^4 + 150a_6 k^2 + 91a_6}{b_1}}$$  

$$C_0= a_1 + 2 a_2 k + 8a_4 k^3 + 96 a_6 k^5$$  

$$w = a_1 k + a_2 k^2 - a_2 + 3a_4 k^4 - 6 a_4 k^2 - a_4 + 35 a_6 k^6 - 75 a_6 k^4 - 15 a_6 k^2 - a_6$$  

$$a_3 = -4a_4 k - 40 a_6 k^3,\quad a_5 = -6 a_6 k$$

$$b_2 = -(24 a_4*\chi^2 + 360 a_6 \chi^2) k^2 + 840 a_6 \frac{\chi^2}{A_1^4},\quad b_3 = 720 a_6 \frac{\chi^3}{A_1^6}$$  

## Literature
**[1]** *M. Raissi, P. Perdikaris, G.E. Karniadakis* Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, Journal of Computational Physics, Volume 378, 2019, Pages 686-707  

**[2]** *Chenxi Wu, Min Zhu, Qinyang Tan, Yadhu Kartha, Lu Lu* A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks, Computer Methods in Applied Mechanics and Engineering, Volume 403, Part A, 2023, 115671  

**[3]** *Sifan Wang, Shyam Sankaran, Paris Perdikaris* Respecting causality for training physics-informed neural networks, Computer Methods in Applied Mechanics and Engineering, Volume 421, 2024, 116813  

**[4]** *Rafael Bischof, Michael A. Kraus* Multi-Objective Loss Balancing for Physics-Informed Deep Learning, [https://arxiv.org/abs/2110.09813](https://arxiv.org/abs/2110.09813)  

**[5]** *Pratik Rathore, Weimu Lei, Zachary Frangella, Lu Lu, Madeleine Udell* Challenges in Training PINNs: A Loss Landscape Perspective, [https://arxiv.org/abs/2402.01868](https://arxiv.org/abs/2402.01868)  

**[6]** *V.A. Medvedev, N.A. Kudryashov* Numerical study of soliton solutions of the cubic-quintic-septic nonlinear Schrodinger equation, Vestnik MEPhI, 2024, vol. 13, no. 2, pp. 83–96

**[7]** *A.A. Bayramukov, N.A. Kudryashov* Numerical study of the model described by the fourth order generalized nonlinear Schrödinger equation with cubic-quintic-septic-nonic nonlinearity, Journal of Computational and Applied Mathematics, Volume 437, 2024, 115497

**[8]** *N.A. Kudryashov* Method for finding highly dispersive optical solitons of nonlinear differential equations, Optik, Volume 206, 2020, 163550

**[9]** *Junwoo Cho, Seungtae Nam, Hyunmo Yang, Seok-Bae Yun, Youngjoon Hong, Eunbyung Park* Separable Physics-Informed Neural Networks, Advances in Neural Information Processing Systems, 2023
