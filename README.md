# Generative models of neural population responses

MATLAB code for learning some published generative models of experimentally 
recorded neural responses. The learning utilizes a form of Persistent 
Contrastive Divergence. Likelihoods are estimated using annealed importance 
sampling. Responses **s** are assumed to be binary vectors.

**Semiparametric pairwise models** have the form

![Vpairwise](https://github.com/jhumplik/generative-neural-models/blob/master/doc/nonlinear_pairwise.png)

where both the coupling matrix **J** and the nonlinear function V have to be
learned from data. The function V should be specified nonparametrically. For 
details, see:
> Humplik J., Tkačik G. (2016)<br>
> Semiparametric energy-based probabilistic models.<br>
> http://arxiv.org/abs/1605.07371<br>

**K-pairwise models** have the form

![Kpairwise](https://github.com/jhumplik/generative-neural-models/blob/master/doc/Kpairwise.png)

These modeles were introduced in
> Tkačik G., Marre O., Amodei D., Schneidman E., Bialek W., et al. (2014)<br>
> Searching for Collective Behavior in a Large Network of Sensory Neurons.<br>
> PLoS Comput Biol 10(1): e1003408.<br>

The above models are generalizations of **pairwise models** 
(fully visible Boltzmann machines):

![pairwise](https://github.com/jhumplik/generative-neural-models/blob/master/doc/pairwise.png)

In the neuroscience context, these models were studied for example in
> Schneidman E., Berry M. J., II, Segev R., and Bialek W. (2006)<br>
> Weak pairwise correlations imply strongly correlated network states in a neural population.<br>
> Nature, 440(7087):1007–12<br>

**Restricted Boltzmann machines (RBMs) have the form:

![pairwise](https://github.com/jhumplik/generative-neural-models/blob/master/doc/rbm.png)

See the reference below for an example application in neuroscience.
> Köster U., Sohl-Dickstein J., Gray C. M., Olshausen B. A. (2014)<br>
> Modeling Higher-Order Correlations within Cortical Microcolumns.<br>
> PLoS Comput Biol 10(7): e1003684.<br>
