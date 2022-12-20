import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class Fit_Gaussian():
    """Class to fit a Gaussian on a histogram"""
    
    def __init__(self, data, normalized=False):
        
        self.data = data                   # The data to fit the Gaussian on.
        self.normalized = normalized       # Using normalized Gaussian. 


    def Gauss(x, x0, sigma, y0, A):
        """Define a Gaussian"""
        return y0 + A * np.exp(-(x - x0)**2 / (2 * sigma**2))
    

    def Gauss_normalized(x, x0, sigma):
        """Define a normalized gaussian"""
        return (1/np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(x - x0)**2 / (2 * sigma**2))

    # Fit a gaussian on a histogram
    def hist_fitting(self, bins=200, show=False):
        """Get a gaussian fitting of a histogram.
        Parameter:
            data - as numpy array
        Returns: 
            X0, sigma, Y0, A - of the histogram gaussian
            or
            X0, sigma - of the normalized histogram gaussian
        """
        # Make a histogram
        if self.normalized:
            n, bins_ = np.histogram(self.data[np.isfinite(self.data)], bins=bins, density=True)
        else:
            n, bins_ = np.histogram(self.data[np.isfinite(self.data)], bins=bins)

        # Data
        x = np.linspace(bins_.min(),bins_.max(),bins_.shape[0]-1)
        y = n

        # Apply the fitting 
        if self.normalized:
            popt,pcov = curve_fit(Fit_Gaussian.Gauss_normalized, x, y,
                                  p0 = ((x.max()+x.min())/2, (x.max()-x.min())/3),
                                  bounds = ([x.min(), 0], [x.max(), (x.max()-x.min())/2]))
        else:
            popt,pcov = curve_fit(Fit_Gaussian.Gauss, x, y,
                                  p0=((x.max()+x.min())/2, (x.max()-x.min())/3, 0, 1/np.sqrt(2 * np.pi * (x.max()/3)**2)),
                                  bounds=([x.min(), 0, -np.inf, -np.inf], [x.max(), (x.max()-x.min())/2, np.inf, np.inf]))            

        # Display the results
        if show:
            plt.figure(figsize=(10,6))
            
            if self.normalized:
                plt.plot(x, Fit_Gaussian.Gauss_normalized(x, *popt), 'r-', alpha = 0.5,
                         label='Gauss: $x_0$ = {:.4f}, $\sigma$ = {:.4f}'.format(*popt))
            else:
                plt.plot(x, Fit_Gaussian.Gauss(x, *popt), 'r-', alpha = 0.5,
                         label='Gauss: $x_0$ = {:.4f}, $\sigma$ = {:.4f}, $Y_0$ = {:.4f}, A = {:.4f}'.format(*popt))
            
            plt.plot(x, y, 'b+:', alpha = 0.4, label='data')            
            plt.legend()
            plt.title('Histogram Gaussian')
            plt.xlabel('value')
            plt.ylabel('frequency')
            plt.grid(alpha=0.3)

        return popt  #(X0, sigma, Y0, A) or just (X0, sigma) for normalized Gaussian
