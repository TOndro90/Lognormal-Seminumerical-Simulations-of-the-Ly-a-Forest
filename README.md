# Lognormal-Seminumerical-Simulations-of-the-Ly-a-Forest
Code for producing synthetic spectra of the Ly-a forest. If you find this code useful in your research, please consider citing following work: Ondro et al.(arXiv:2412.11909)

## lognormal_tau(cosmo, therm_params, kpk_linear, real, roll, zmin, zmax, npts)
Returns array of float with 7 rows: redshift, $$\delta_{x}, v_{x}, n_{b}, T_{x}, n_{\text{HI}}, \tau$$

## Parameters:
cosmo - Cosmological parameters <br>
therm_params - thermal parameters and Jeans length <br>
kpk_linear - matter power spectrum <br>
real - seed (for repeatability) <br>
roll - parameter for application of periodic boundary conditions (in percentage of total spectrum length) <br>
zmin - minimum redshift <br>
zmax - maximum redshift <br>
npts - number of points <br>

## Example:
############################# Thermal parameters ############################## <br>
params = {} <br>
params['xJ'] = 0.12                # in Mpc units <br>
params['logT0'] = np.log10(10000) <br>
params['gamma'] = 1.20 <br>
params['Gamma'] = 1.346e-12       <br>
params['nu'] = 1.  <br>
########################## Cosmological parameters ############################ <br>
cosmology = {} <br>
cosmology['omega_b'] = 0.0456 <br>
cosmology['omega_M'] = 0.0456 + 0.227 <br>
cosmology['omega_L'] = 0.728 <br>
cosmology['omega_k'] = 0 <br>
cosmology['h'] = 0.704 <br>
cosmology['n'] = 0.996 <br>
cosmology['sigma_8'] = 0.809 <br>
cosmology['tau'] = 0.087 <br>
cosmology['z_reion'] = 10.4 <br>
cosmology['t_0'] = 13.75 <br>
cosmology['Y_He'] = 0.24 <br>
########################## Matter power spectrum ############################## <br>
kpk_linear = np.loadtxt("pk_linear_camb.txt") <br>
############################################################################### <br>
a = lognormal_tau(cosmology, params, kpk_linear, 5, 0.05, 2.45, 2.55, 2048) <br>
