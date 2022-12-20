# GUV-SLB adhesion system under Reflection Interference Contrast Microscopy (RICM)
This project has the following:
* Script contains the model to compute the height of a GUV adhered to an SLB.
* Note-book contains an example to compute the height of GUV adhered to an SLB mediated by E-cadherin.
* Utilities script contains a function to help fitting a Gaussian on the resulted histogram.
## Determination of intensity vs height relation
The reflectivity of $l$ successive layers is given by
$$R_l = r_{01} + \sum_{i=1}^l \left[\prod_{s=1}^i (1-r^2_{s-1,s}) \exp{(-ik\Delta_s)}\right] r_{i, i+1}$$
Such that:
$$r_{ij} = \frac{n_i - n_j}{n_i + n_j} \ , \ k = \frac{2 \pi}{\lambda} \ , \ \Delta_i = 2 n_i d_i$$
Where $n_i$ and $di$ are refractive index and the thickness of the layer $i$ respectively, and $\lambda$ is the wave length of the incident light.
Intensity observed from $l$ layers is given by 
$$I(x,y) = R_l^* R_l I_0$$
The intensity is calculated for the each value of height $h$ as from 0 to 200 nm, normalized with respect to the background and plotted against $h$
$$I_{norm}(h) = y_0 - A \cos{\left(\frac{4 \pi n_{out}}{\lambda} (h - h_0) + \phi \right)}$$
Where $y_0$ , $A$ and $h_0$ are fitting parameters and $\phi$ is the phase shift.\\
Inverting the last, we get $h$ as a function of $I_{norm}$ and the parameters determined from the fitting
$$h = \frac{\lambda}{4 \pi n_{out}} \left[\arccos{\left(\frac{y_0 - I_{norm}}{A} \right)} - \phi \right] + h_0$$
Applying this on the normalized image gives the height image.
