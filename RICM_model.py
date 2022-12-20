# Libraries
import numpy as np
import matplotlib.pyplot as plt
from skimage import restoration, filters, morphology
from scipy import ndimage,linalg, optimize
from Utilities import Fit_Gaussian


class Height_map():
    """This class is used to get the dependence of the normalized intensity on the height for GUV-SLB adhesion system."""
    
    def __init__(self,
                 n_glass = 1.525,
                 n_water = 1.333,
                 n_outer = 1.335,
                 n_lipid = 1.486,
                 n_inner = 1.344,
                 d_water = 1,
                 d_lipid = 4,
                 l       = 546,
                 p       = 0):
       
        # Parameters
        self.n_glass = n_glass      # refractive index of glass 
        self.n_water = n_water      # refractive index of water
        self.n_outer = n_outer      # refractive index of outer solution (PBS)
        self.n_lipid = n_lipid      # refractive index of lipid
        self.n_inner = n_inner      # refractive index of inner buffer (Sucrose)
        self.d_water = d_water      # thickness of water in nm
        self.d_lipid = d_lipid      # thickness of lipid in nm
        self.l       = l            # wave length of the RICM light in nm
        self.p       = p            # phase shift of the cosine function
        

    def i5_norm(self, h):
        """Normalized reflectance for 5 interfaces"""

        # Wave vector
        k = (2 * np.pi) / self.l

        # Refractive indices
        n0 = self.n_glass    # Glass slid
        n1 = self.n_water    # Layer of water down the SLB
        n2 = self.n_lipid    # SLB 
        n3 = self.n_outer    # Outer buffer PBS
        n4 = self.n_lipid    # Vesicle membrane
        n5 = self.n_inner    # Inner buffer Sucrose

        # Fresnel reflection coefficients
        r01 = (n0-n1) / (n0+n1)
        r12 = (n1-n2) / (n1+n2)
        r23 = (n2-n3) / (n2+n3)
        r34 = (n3-n4) / (n3+n4)
        r45 = (n4-n5) / (n4+n5)

        # Distances traveled by light
        D1 = 2 * n1 * self.d_water
        D2 = 2 * n2 * self.d_lipid
        D3 = 2 * n3 * h
        D4 = 2 * n4 * self.d_lipid  
        
        # Effective reflection coefficient
        R1 = r01
        R2 = ((1-r01**2) * np.exp(-1j*k*D1)) * r12
        R3 = ((1-r01**2)*(1-r12**2) * np.exp(-1j*k*(D1+D2))) * r23
        R4 = ((1-r01**2)*(1-r12**2)*(1-r23**2) * np.exp(-1j*k*(D1+D2+D3))) * r34
        R5 = ((1-r01**2)*(1-r12**2)*(1-r23**2)*(1-r34**2) * np.exp(-1j*k*(D1+D2+D3+D4))) * r45
        
        # Effective reflection coefficient of the adhesion zone
        R = R1 + R2 + R3 + R4 + R5

        # Effective reflection coefficient of the background
        R_b = R1 + R2 + R3

        # Normalized reflectance R_norm
        R_norm = (np.abs(R * np.conjugate(R)) - np.abs(R_b * np.conjugate(R_b))) / np.abs(R_b * np.conjugate(R_b))

        return R_norm

    
    def normalized_intensity(self, h, Y0, A, h0):
        """The dependence of the normalized intensity on height"""
        
        n_outer = self.n_outer  #refractive index of PBS
        l = self.l              #wave length of the RICM light

        return Y0 - A * np.cos((4 * np.pi * n_outer / l) * (h - h0) + 2*np.pi*self.p)
    


class RICM(Height_map):
    """Treating the RICM image of GUV-SLB adhesion system to get the height of the GUV"""
    
    def __init__(self,
                 img,
                 denoise=True, nl_fast_mode=True, nl_patch_size=10, nl_patch_distance=1,
                 hole=3, remove_small=True, min_size=64,
                 n_glass=1.525, n_water=1.333, n_outer=1.335, n_lipid=1.486, n_inner=1.344,
                 d_water=1, d_lipid=4, l=546, p=0):
        
        # The image
        self.img = img
        
        # RICM parameters
        self.n_glass = n_glass      # refractive index of glass 
        self.n_water = n_water      # refractive index of water
        self.n_outer = n_outer      # refractive index of outer solution (PBS)
        self.n_lipid = n_lipid      # refractive index of lipid
        self.n_inner = n_inner      # refractive index of inner buffer (Sucrose)
        self.d_water = d_water      # thickness of water in nm
        self.d_lipid = d_lipid      # thickness of lipid in nm
        self.l = l                  # wave length of the RICM light in nm
        self.p = p                  # phase shift of the cosine function

        # Denoising parameters
        self.denoise = denoise
        self.nl_fast_mode = nl_fast_mode
        self.nl_patch_size = nl_patch_size
        self.nl_patch_distance = nl_patch_distance
        
        # Mask parameters
        self.hole = hole                     # hole filling kernel
        self.remove_small = remove_small     # remove small defects
        self.min_size = min_size             # minimum size for a small defect
          

    def nl_denoise(self):
        """Denoise the image using Non-local means denoising algorithm"""
        # Apply the Non-local means denoising algorithm and return the denoised image
        return restoration.denoise_nl_means(self.img,
                                            h = np.mean(restoration.estimate_sigma(self.img)),
                                            fast_mode = self.nl_fast_mode,
                                            patch_size = self.nl_patch_size,
                                            patch_distance = self.nl_patch_distance)
        


    def edge_detection(self):
        """Detecting the edges"""
        # Check if denoising is True
        if self.denoise:
            # Return the edge of the denoised image
            return filters.sobel(self.nl_denoise())
        else:
            # Return the edge of the original image
            return filters.sobel(self.img)


    def mask(self):
        """Determine the contact zone by filling the closed edges inside the binary image of the edges"""

        #Applying some edge operators to the denoised image
        edge = self.edge_detection()
        
        # 1- Getting the threshold of edge filtered image
        # 2- Making a binary image with 0 and 1 values
        # 3- Fill the detected edge
        # 4- Remove the small objects in case of remove small is True
        
        if self.remove_small: 
            return morphology.remove_small_objects(ndimage.binary_fill_holes(np.multiply(edge > filters.threshold_otsu(edge), 1),
                                                                             structure=np.ones((self.hole, self.hole))),
                                                   min_size=self.min_size)

        return ndimage.binary_fill_holes(np.multiply(edge > filters.threshold_otsu(edge), 1),
                                         structure=np.ones((self.hole, self.hole)))


    def background_fitting(self):
        """Fitting the background"""

        # Determine the contact zone
        edge_binary_filled = self.mask()

        # Write the data in terms of 3-dim points excluding the contact zone
        coord_background_intensity = []
        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):
                if edge_binary_filled[i, j] == False:  # excluding the contact zone
                    coord_background_intensity.append([i, j, self.img[i,j]])

        # 3-dim data points
        data = np.array(coord_background_intensity)

        # best-fit quadratic curve
        A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
        C,_,_,_ = linalg.lstsq(A, data[:,2])

        # Copy the original img
        Background = np.ones(self.img.shape)

        # Fill the background with the values came from the fitting
        for X in range(self.img.shape[0]):
            for Y in range(self.img.shape[1]):
                Background[X,Y] = C[0] + C[1]*X + C[2]*Y + C[3]*X*Y + C[4]*X*X + C[5]*Y*Y

        return Background


    def correct(self):
        """Correcting the image by subtracting the background then add it's average to each pixel"""

        # Fitting the background
        Background = self.background_fitting()
        
        # Return the corrected image
        return self.img - Background + np.average(Background)

    
    def background_normalization(self):
        """Normalized reflectance to the background"""

        # Get the corrected image
        img_corrected = self.correct()

        # Get the background by removing the contact zone from the corrected image
        corrected_background = img_corrected * (1 - self.mask())

        # Transform it into a histogram excluding the contact zone
        corrected_background = corrected_background[corrected_background != 0].ravel()
        
        # Fit a Gaussian on the corrected background histogram then take it's mean
        avg_corrected_background, _ = Fit_Gaussian(corrected_background, normalized=True).hist_fitting()

        return (img_corrected - avg_corrected_background) / avg_corrected_background
    
    
    def height(self, h=np.linspace(1, 600, 600)):
        """RICM height mapping"""
        
        # Fit the parameters Y_0, A, h_0 of the cosine function
        mapping = Height_map(n_glass = self.n_glass,
                             n_water = self.n_water,
                             n_outer = self.n_outer,
                             n_lipid = self.n_lipid,
                             n_inner = self.n_inner,
                             d_water = self.d_water,
                             d_lipid = self.d_lipid,
                             l       = self.l,
                             p       = self.p)
        
        popt, _ = optimize.curve_fit(mapping.normalized_intensity, h, mapping.i5_norm(h))
        Y0, A, h0 = popt
        print('Y0 = {:.2f}, A = {:.2f}, h0 = {:.2f}'.format(*popt))

        return (self.l/(4*np.pi*self.n_outer)) * (np.arccos((Y0 - self.background_normalization()) / A) - 2*np.pi*self.p) + h0    
    
    
    def height_argument(self, h=np.linspace(1, 600, 600)):
        """RICM height mapping argument"""

        # Fit the parameters Y_0, A, h_0 of the cosine function
        mapping = Height_map(n_glass = self.n_glass,
                             n_water = self.n_water,
                             n_outer = self.n_outer,
                             n_lipid = self.n_lipid,
                             n_inner = self.n_inner,
                             d_water = self.d_water,
                             d_lipid = self.d_lipid,
                             l       = self.l,
                             p       = self.p)
        
        popt, _ = optimize.curve_fit(mapping.normalized_intensity, h, mapping.i5_norm(h))
        Y0, A, _ = popt

        return (Y0 - self.background_normalization()) / A 
    
    
    def show_summary(self, name='summary', save=False):
        """Display the way to the RICM height mapping step by step"""
        
        plt.figure(figsize=(23,7))

        plt.subplot(251)
        plt.axis('off')
        plt.title('Orginal image')
        plt.imshow(self.img, cmap = "gray")
        plt.colorbar()

        plt.subplot(252)
        plt.axis('off')
        plt.title('Denoised image')
        plt.imshow(self.nl_denoise() , cmap = 'gray')
        plt.colorbar()

        plt.subplot(253)
        plt.axis('off')
        plt.title('Edge detected image')
        plt.imshow(self.edge_detection() , cmap = 'gray')
        plt.colorbar();

        plt.subplot(254)
        plt.axis('off')
        plt.title('Masked image')
        plt.imshow(self.mask() , cmap = 'gray')
        plt.colorbar();

        plt.subplot(255)
        plt.axis('off')
        plt.title('Background fitted image')
        plt.imshow(self.background_fitting() , cmap = 'gray')
        plt.colorbar();

        plt.subplot(256)
        plt.axis('off')
        plt.title('Corrected image')
        plt.imshow(self.correct() , cmap = 'gray')
        plt.colorbar()

        plt.subplot(257)
        plt.axis('off')
        plt.title('Background normalized image')
        plt.imshow(self.background_normalization() , cmap = 'gray')
        plt.colorbar()
        
        # Show the argument of the arccosine to make sure it's between 1 and -1
        plt.subplot(258)
        plt.axis('off')
        plt.title('Arccosine argument image')
        plt.imshow(self.height_argument() , cmap = 'inferno')
        plt.colorbar();
        
        plt.subplot(259)
        plt.axis('off')
        plt.title('Height image')
        plt.imshow(self.height() , cmap = 'inferno')
        plt.colorbar()
        
        plt.subplot(2,5,10)
        plt.title('Height histogram')
        plt.xlabel('$h_{[nm]}$')
        #plt.ylabel('Frequency')
        plt.hist(self.height().ravel(), bins = 200)
        plt.grid(alpha=0.2)
        
        # Save the image
        if save: plt.savefig(name)

        # Show the results
        plt.show()

        
    def model_fitting(self, name='diaphragm_summary', h=np.linspace(-100, 200, 600), show=False, save=False):
        """Extract the height on the contact zone"""

        # Define the Normalized intensity, height only on the contact zone using the mask
        mask     = RICM.mask(self)
        i_norm   = RICM.background_normalization(self)
        i_height = RICM.height(self, h)            

        # Remove the masked values without removing the original zeros
        i_norm[~mask]   = np.nan
        i_height[~mask] = np.nan
        
        if show:
            # Plot the model vs data
            plt.figure(figsize=(20,5))

            # The histogram of the normalized image
            plt.subplot(121)
            plt.hist(RICM.background_normalization(self).ravel(), bins=200, label='Whole Image', alpha=0.5)
            plt.hist(i_norm.ravel(), bins=200, color='r', alpha=0.5, label='Contact Zone')
            plt.title('Normalized image histogram', fontsize='x-large')
            plt.xlabel('$I_{norm}$', fontsize='x-large')
            plt.ylabel('Frequency', fontsize='x-large')
            plt.legend(fontsize='x-large')
            plt.grid(alpha=0.2)

            plt.subplot(122)
            plt.plot(h, RICM.i5_norm(self, h), '--', label='Model $I_{norm}\ (h)$')
            plt.scatter(i_height, i_norm, color='r', alpha=0.3, label='Contact Zone Intensity')
            plt.title(f'$I_n(h)$ for $\lambda$ = {self.l}, n_inner = {self.n_inner} and $\phi$ = {self.p}', fontsize='x-large')
            plt.xlabel('$h_{[nm]}$', fontsize='x-large')
            plt.ylabel('$I_{norm}$', fontsize='x-large')
            plt.legend(fontsize='x-large')
            plt.grid(alpha=0.2)

            # Save the image
            if save:plt.savefig(name)

            # Show the results
            plt.show()
            
        return i_height, i_norm
