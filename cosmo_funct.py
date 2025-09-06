# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 12:39:16 2024

@author: Tomas
"""
import numpy
import scipy.integrate as si

Mpc_cm = 3.08568025e24 # cm
Mpc_km = Mpc_cm * 1.0e-5 # km
H100_s = 100. / Mpc_km # s^-1
c_light_cm_s = 29979245800. # cm/s
c_light_Mpc_s = c_light_cm_s / Mpc_cm # Mpc / s
M_sun_g = 1.98892e33 # g
amu_g = 1.66053886e-24 #g
m_H_g = 1.00794 * amu_g # g
m_He_g = 4.002602 * amu_g # g
G_const_Mpc_Msun_s = M_sun_g * (6.673e-8) / Mpc_cm**3. # Mpc^3 msun^-1 s^-2 
pi = numpy.pi


### Distance ###
def e_z(z, **cosmo):
    """The unitless Hubble expansion rate at redshift z.

    In David Hogg's (arXiv:astro-ph/9905116v4) formalism, this is
    equivalent to E(z), defined in his eq. 14.

    Modified (JBJ, 29-Feb-2012) to include scalar w parameter

    """

    if 'w' in cosmo:
        #ow = exp( si.quad( lambda zp: (1.+cosmo['w']) / (1.+zp), 0., z, limit=1000 ) )
        return (cosmo['omega_M'] * (1+z)**3. + 
                cosmo['omega_k'] * (1+z)**2. + 
                cosmo['omega_L'] * (1+z)**(1+cosmo['w']) )**0.5
    else:
        return (cosmo['omega_M'] * (1+z)**3. + 
                cosmo['omega_k'] * (1+z)**2. + 
                cosmo['omega_L'])**0.5

def hubble_z(z, **cosmo):
    """The value of the Hubble constant at redshift z.

    Units are s^-1

    In David Hogg's (arXiv:astro-ph/9905116v4) formalism, this is
    equivalent to H_0 * E(z) (see his eq. 14).

    """
    H_0 = cosmo['h'] * H100_s

    return H_0 * e_z(z, **cosmo)

def _comoving_integrand(z, omega_M, omega_L, omega_k, h, w=-1.):

    e_z = (omega_M * (1+z)**3. + 
           omega_k * (1+z)**2. + 
           omega_L * (1+z)**(1.+w))**0.5
    
    H_0 = h * H100_s
    
    H_z =  H_0 * e_z

    return c_light_Mpc_s / (H_z)

def comoving_distance(z, z0 = 0, **cosmo):
    """The line-of-sight comoving distance (in Mpc) to redshift z.

    See equation 15 of David Hogg's arXiv:astro-ph/9905116v4

    Units are Mpc.

    Optionally calculate the integral from z0 to z.

    Returns
    -------
    
    d_co: ndarray
       Comoving distance in Mpc.

    Examples
    --------

    >>> import cosmolopy.distance as cd
    >>> cosmo = {'omega_M_0' : 0.3, 'omega_lambda_0' : 0.7, 'h' : 0.72}
    >>> cosmo = cd.set_omega_k_0(cosmo)
    >>> d_co = cd.comoving_distance(6., **cosmo)
    >>> print "Comoving distance to z=6 is %.1f Mpc" % (d_co)
    Comoving distance to z=6 is 8017.8 Mpc

    """

    #cosmo = set_omega_k_0(cosmo)

    if 'w' in cosmo:
        w = cosmo['w']
    else:
        w = -1.

    dc_func = \
        numpy.vectorize(lambda z, z0, omega_M, omega_L, omega_k, h, w: 
                        si.quad(_comoving_integrand, z0, z, limit=1000,
                                args=(omega_M, omega_L, omega_k, h, w)))
    d_co, err = dc_func(z, z0, cosmo['omega_M'],
                        cosmo['omega_L'],
                        cosmo['omega_k'],
                        cosmo['h'],
                        w
                        )
    return d_co


### Density ###
def get_omega_k_0(**cosmo):
    """'Spatial curvature density' omega_k_0 for a cosmology (if needed).

    If omega_k_0 is specified, return it. Otherwise return:

      1.0 - omega_M_0 - omega_lambda_0

    """
 
    if 'omega_k' in cosmo:
        omega_k_0 = cosmo['omega_k']
    else:
        omega_k_0 = 1. - cosmo['omega_M'] - cosmo['omega_L']
    return omega_k_0

def omega_M_z(z, **cosmo):
    """Matter density omega_M as a function of redshift z.

    Notes
    -----

    From Lahav et al. (1991, MNRAS 251, 128) equations 11b-c. This is
    equivalent to equation 10 of Eisenstein & Hu (1999 ApJ 511 5).

    """
    if get_omega_k_0(**cosmo) == 0:
        return 1.0 / (1. + (1. - cosmo['omega_M'])/
                      (cosmo['omega_M'] * (1. + z)**3.))
    else:
        return (cosmo['omega_M'] * (1. + z)**3. / 
                e_z(z, **cosmo)**2.)


def get_X_Y(**cosmo):
    """The fraction of baryonic mass in hydrogen and helium.

    Assumes X_H + Y_He = 1.

    You must specify either 'X_H', or 'Y_He', or both.
    """
    if 'X_H' in cosmo and 'Y_He' not in cosmo:
        X_H = cosmo['X_H']
        Y_He = 1. - X_H
    elif 'Y_He' in cosmo and 'X_H' not in cosmo:
        Y_He = cosmo['Y_He']
        X_H = 1. - Y_He
    else:
        X_H = cosmo['X_H']
        Y_He = cosmo['Y_He']
    return X_H, Y_He

def cosmo_densities(**cosmo):
    """The critical and mean densities of the universe.

    Returns
    -------
    rho_crit and rho_0 in solar masses per cubic Megaparsec.

    """

    omega_M_0 = cosmo['omega_M']
    h = cosmo['h']

    rho_crit = 3. * (h * H100_s)**2. / (8. * pi * G_const_Mpc_Msun_s)
    rho_0 = omega_M_0 * rho_crit
    
    #print " Critical density rho_crit = %.3g Msun/Mpc^3" % rho_crit
    #print " Matter density      rho_0 = %.3g Msun/Mpc^3" % rho_0

    return rho_crit, rho_0

def baryon_densities(**cosmo):
    """Hydrogen number density at z=0.

    Parameters
    ----------

    cosmo: cosmological parameters

    parameters used: 'omega_b_0', 'X_H' and/or 'Y_He', plus those
    needed by cosmo_densities.
       

    Returns
    -------

    rho_crit, rho_0, n_He_0, n_H_0

    The first two are in units of solar masses per cubic
    Megaparsec. The later two are in number per cubic Megaparsec.
    
    """
    
    X_H, Y_He = get_X_Y(**cosmo)

    rho_crit, rho_0 = cosmo_densities(**cosmo)

    n_H_0 = (rho_crit * cosmo['omega_b'] * X_H * M_sun_g / 
             m_H_g)
    n_He_0 = (rho_crit * cosmo['omega_b'] * Y_He * M_sun_g / 
              m_He_g)
    #    print " Hydrogen number density n_H_0 = %.4g (Mpc^-3)" % n_H_0
    return rho_crit, rho_0, n_He_0, n_H_0






