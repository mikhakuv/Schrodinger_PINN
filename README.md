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

The idea was proposed in [2]. To enable this option, set `PINN.points_gen_method` as `"first"`,`"second"` or `"third"` correspondingly. By default `PINN.points_gen_method = "random"`  
### Causal Loss  
By default, loss on equation is calculated as follows:  

$$Loss_{eq} = \frac{1}{N_t}\sum_{k=1}^{N_t} Loss_{eq}(t_k),\ \text{where}\ Loss_{eq}(t_k)\ \text{is mean residual on points with}\ t=t_k$$  

Causal loss employs differrent formula which represents the idea of causality in training suggested in [3]:  

$$Loss_{eq} = \frac{1}{N_t\cdot\sum_{k=1}^{N_t}w_k}\sum_{k=1}^{N_t} Loss_{eq}(t_k)\cdot w_k$$  

$$\text{where}\ w_k = exp\left(-\varepsilon\cdot\sum_{i=1}^{k-1}Loss_{eq}(t_k)\right)\ \text{and}\ \varepsilon\ \text{is changeable parameter}$$  

Causality means that if $Loss_{eq}(t)$ is high for $t<\tilde{t}$, then PINN will not be training on $t>\tilde{t}$ until $Loss_{eq}(t)$ for $t<\tilde{t}$ gets lower.  
To enable this option, set `PINN.causal_loss = True` and adjust `PINN.epsilon`.  

### Loss Balancing  
Loss for PINN is defined as follows:  

$$Loss = \lambda_i\cdot Loss_{ic} + \lambda_b\cdot Loss_{bc} + \lambda_f\cdot Loss_{eq}$$  

Without active loss balancing, $\lambda_i,\ \lambda_b, \lambda_f$ are constant values, set as `PINN.lambda_i=10/12`, `PINN.lambda_b=1/12`, `PINN.lambda_f=1/12` by default. However, active loss balancing option is also available, namely ReloBRaLo method which is introduced in [4]. It changes $\lambda_i,\ \lambda_b, \lambda_f$ dinamically according to the following formula:  

$$\lambda_n = \tau\cdot\left(\rho\cdot\lambda_n(iter-1) + (1-\rho)\cdot\widehat{\lambda_n}(iter)\right) + (1-\tau)\cdot\lambda_n(iter)$$  

$$\text{where}\ \lambda_n(iter) = \frac{exp\left(\frac{loss_n(iter)}{loss_n(iter-1)}\right)}{\sum\limits_{k} exp\left(\frac{loss_k(iter)}{loss_k(iter-1)}\right)},\ \widehat{\lambda_n}(iter) = \frac{exp\left(\frac{loss_n(iter)}{loss_n(0)}\right)}{\sum\limits_{k} exp\left(\frac{loss_k(iter)}{loss_k(0)}\right)},$$  

$$\tau \in[0,1] - \text{extinction coefficient},\ \rho \in[0,1] - \text{random lookback coefficient}$$

Set `PINN.loss_bal_method="relobralo"` for using active loss balancing method. By default `PINN.loss_bal_method="none"`.
### Optimizers  
Three optimizers are available: ADAM, LBFGS and NNCG (see [5] for details). Amount of steps by each can be set using `PINN.adam_steps`, `PINN.lbfgs_steps` and `PINN.nncg_steps`. Moreover, exponential step decay can be established by `PINN.adam_step_decay`, `PINN.lbfgs_step_decay` `PINN.nncg_step_decay` combined with `PINN.decay_freq` which defines frequency of decay.  

## Plots and Statistics
### Metrics
For accuracy evaluation following metrics are used:  

$$Rel_h= \frac{\sqrt{\sum_{n=1}^{N} (|\hat{q_{n}}| - |q_{n}|)^2}}{\sqrt{\sum_{i=1}^{N} (q_{i})^2}}$$  

$$max(Lw_1) = \max_{i \in T} \left ( \frac{|I_1 - \hat{I}_1(t_i)|}{I_1} \right ) \cdot 100\\%$$  

$$mean(Lw_1) = \text{mean}_{i \in T} \left ( \frac{|I_1 - \hat{I}_1(t_i)|}{I_1} \right ) \cdot 100\\%$$  

$$max(Lw_2) = \max_{i \in T} \left (\frac{|I_2 - \hat{I}_2(t_i)|}{I_2} \right ) \cdot 100\\%$$  

$$mean(Lw_2) = \text{mean}_{i \in T} \left (\frac{|I_2 - \hat{I}_2(t_i)|}{I_2} \right ) \cdot 100\\%$$  

$$\text{where hatted values are predicted values and unhatted values are ground truth}$$  

Plenty visualizing options are available:  
### Training History
`PINN.train_hist(logscale=bool, step=int)` returns loss history with `step` intervals using or not using logscale:  

<img src="https://github.com/mikhakuv/Schrodinger_PINN/blob/main/pictures/train_hist.png">  

### Residual of Solution
`PINN.plot_residual(X=np.ndarray, T=np.ndarray))` plots residual of obtained solution:  

<img src="https://github.com/mikhakuv/Schrodinger_PINN/blob/main/pictures/plot_residual.png">  

### Other Plots
`plot_comparison(X=np.ndarray, T=np.ndarray, Q_pred=np.ndarray, Q_truth=np.ndarray, savefig=bool, namefig=string)` shows comparison of `Q_pred` and `Q_truth`:  

<img src="https://github.com/mikhakuv/Schrodinger_PINN/blob/main/pictures/plot_comparison.png">  

`plot_errors(X=np.ndarray, T=np.ndarray, Q_pred=np.ndarray, Q_truth=np.ndarray, savefig=bool, namefig=string, savetable=bool, nametable=string)` uses `Q_pred` and `Q_truth` to plot $|q(x,t)|$, $Lw_1(t)$, $Lw_2(t)$ and $Rel_h(t)$ and return dictionary with evaluated metrics:  

<img src="https://github.com/mikhakuv/Schrodinger_PINN/blob/main/pictures/plot_errors.png">  

## Problems  
Two problems are given as an example:
### 2nd order  
equation:  

$$iq_t + q_{xx} + |q|^2 q (1 - \alpha |q|^2 + \beta |q|^4) = 0$$  

$$\alpha=1,\quad \beta=0$$  

solution:  

$$q(x,t)=\frac{(k^2-w)e^{\sqrt{k^2-w}\cdot (x-2kt-x_0)}}{\frac{1}{16}+2(k^2-w)\cdot e^{2\cdot\sqrt{k^2-w}\cdot (x-2kt-x_0)}}\cdot e^{i(kx-wt+\theta_0)}$$  

$$\text{where}\ k,\ w,\ x_0,\ \theta_0\ \text{are changeable parameters}$$  

### 6th order  
equation:  

$$iq_t + ia_1q_x + a_2q_{xx} + ia_3q_{3x} + a_4q_{4x} + ia_5q_{5x} + a_6q_{6x} + q(b_1|q|^2 +b_2|q|^4 + b_3|q|^6)=0$$  

solution:  

$$q(x,t) = \frac{A_1}{a e^{(x-C_0 t + x_0)} + \frac{\chi}{4a}\cdot e^{-(x-C_0 t + x_0)}}\cdot e^{i(kx-wt+\theta_0)}$$

$$\text{where}\ a_1,\ a_2,\ a_4,\ a_6,\ b_1,\ \chi,\ a,\ x_0,\ \theta_0 - \text{changeable, while other parameters are found using following expressions:}$$

$$k=1$$  

$$A_1 = \sqrt{2\chi\cdot\frac{a_2-6a_4 k^2 + 12a_4 k + 10a_4 + 75a_6 k^4 + 150a_6 k^2 + 91a_6}{b_1}}$$  

$$C_0= a_1 + 2 a_2 k + 8a_4 k^3 + 96 a_6 k^5$$  

$$w = a_1 k + a_2 k^2 - a_2 + 3a_4 k^4 - 6 a_4 k^2 - a_4 + 35 a_6 k^6 - 75 a_6 k^4 - 15 a_6 k^2 - a_6$$  

$$a_3 = -4a_4 k - 40 a_6 k^3,\quad a_5 = -6 a_6 k$$

$$b_2 = -(24 a_4*\chi^2 + 360 a_6 \chi^2) k^2 + 840 a_6 \frac{\chi^2}{A_1^4},\quad b_3 = 720 a_6 \frac{\chi^3}{A_1^6}$$  

## Literature
**1** *M. Raissi, P. Perdikaris, G.E. Karniadakis* Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, Journal of Computational Physics, Volume 378, 2019, Pages 686-707  
**2** *Chenxi Wu, Min Zhu, Qinyang Tan, Yadhu Kartha, Lu Lu* A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks, Computer Methods in Applied Mechanics and Engineering, Volume 403, Part A, 2023, 115671  

**3** *Sifan Wang, Shyam Sankaran, Paris Perdikaris* Respecting causality for training physics-informed neural networks, Computer Methods in Applied Mechanics and Engineering, Volume 421, 2024, 116813  

**4** *Rafael Bischof, Michael A. Kraus* Multi-Objective Loss Balancing for Physics-Informed Deep Learning, [https://arxiv.org/abs/2110.09813](https://arxiv.org/abs/2110.09813)  

**5** *Pratik Rathore, Weimu Lei, Zachary Frangella, Lu Lu, Madeleine Udell* Challenges in Training PINNs: A Loss Landscape Perspective, [https://arxiv.org/abs/2402.01868](https://arxiv.org/abs/2402.01868)  
