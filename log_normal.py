import numpy as np
import math
import cosmo_funct as cosmo_functions
import scipy as sp

# CGS units are default
c_km_s = 299792.45800
H100_s = 3.24077648681e-18
Mpc = 3.08568025e+24
km = 1.e5
kboltz=1.3806504e-16
mprot = 1.6726231e-24
i_alpha=4.45e-18
damp_gamma = 6.409021872e-3  ## km/s


def round_up_to_even(f):
    return math.ceil(f / 2.) * 2                                               


def voigt_wofz(a, u):
    try:
         from scipy.special import wofz
    except ImportError:
         s = ("Can't find scipy.special.wofz(), can only calculate Voigt "
              " function for 0 < a < 0.1 (a=%g)" % a)  
         print (s)
    else:
         return wofz(u + 1j * a).real

def d_plus(z, cdict):

    a_upper = 1.0 / (1.0 + z)
    lna = np.linspace(np.log(1e-8), np.log(a_upper), 1000)
    z_vec = 1.0 / np.exp(lna) - 1.0

    integrand = 1.0 / (np.exp(lna) * cosmo_functions.e_z(z_vec, **cdict)) ** 3

    integral = sp.integrate.simpson(np.exp(lna) * integrand, dx=lna[1] - lna[0])
    dplus = 5.0 * cdict["omega_M"] * cosmo_functions.e_z(z, **cdict) * integral / 2.0

    return dplus

def growth_factor(z, cdict):
    growth = d_plus(z, cdict) / d_plus(0.0, cdict)
    return growth

def lognormal_tau(cosmo, therm_params, kpk_linear, real, roll, zmin, zmax, npts):
    lnkk_org = np.log(kpk_linear[:,0])
    lnpk_org = np.log(kpk_linear[:,1])
    lnkk = np.linspace(-9, 5, 10000)
    lnpk = np.interp(lnkk, lnkk_org, lnpk_org)                                  # interpolate input power spectrum
    z = 0.5*(zmin + zmax)
    lnkk = lnkk + np.log(cosmo['h'])                                            # convert from h/Mpc to 1/Mpc
    lnpk = lnpk - 3 * np.log(cosmo['h'])                                        # convert from (Mpc/h)^3 to Mpc^3
    Dz = growth_factor(z, cosmo)
    
    pk_b_z = np.exp(lnpk + 2 * np.log(Dz)) * np.exp(-2 * therm_params['xJ'] ** 2 * np.exp(lnkk) ** 2)
    pk_b_z += 1e-100                                                            # adding a very small number to avoid zeros
    boxsize_z = zmax - zmin
    boxsize = cosmo_functions.comoving_distance(zmax, **cosmo) - cosmo_functions.comoving_distance(zmin, **cosmo)
    npix = npts
    
    dx = boxsize / npix
    kval = np.fft.fftfreq(npix, d=dx/(2*np.pi))
    lnkval = np.log(np.abs(kval)+1.e-8)
    zval = np.linspace(zmin, zmax, num=npix, endpoint=False)
    
   
    integrand_pk_times_k = pk_b_z * np.exp(2 * lnkk)                            
    integrand_pk_by_k = pk_b_z
    integrand_pk_by_kcube = pk_b_z * np.exp(- 2 * lnkk)

    def integral_interpolated(integrand_fk, lnkk, lnkval):
        integral_fk = np.cumsum(integrand_fk[::-1])[::-1]
        integral_interpolated = np.interp(lnkval, lnkk, integral_fk)
        return integral_interpolated * (lnkk[1] - lnkk[0])

    integral_pk_times_k = integral_interpolated(integrand_pk_times_k, lnkk, lnkval)
    integral_pk_by_k = integral_interpolated(integrand_pk_by_k, lnkk, lnkval)
    integral_pk_by_kcube = integral_interpolated(integrand_pk_by_kcube, lnkk, lnkval)

    beta_k = integral_pk_by_kcube / integral_pk_by_k

    pwk = (0.5 / np.pi) * integral_pk_by_k / beta_k
    puk = (0.5 / np.pi) * integral_pk_times_k - pwk
    tiny1 = 0.01 * np.min(pwk[pwk > 0])                                         # adding tiny values to avoid nans
    tiny2 = 0.01 * np.min(puk[puk > 0])
    pwk += tiny1
    puk += tiny2

    seedtable = np.linspace(121270, 171269, 50000, dtype=int)
    rng = np.random.RandomState(seed=seedtable[real])
    random_1 = rng.normal(0.0, 1.0, npix)
    random_2 = rng.normal(0.0, 1.0, npix)
    
    w_re = np.zeros([npix])
    w_re[1:npix//2] = random_1[1:npix//2] * np.sqrt(pwk[1:npix//2] / 2)
    w_re[npix//2+1:npix] = w_re[npix//2-1:0:-1]

    w_im = np.zeros([npix])
    w_im[1:npix//2] = random_1[npix//2:npix-1] * np.sqrt(pwk[1:npix//2] / 2)
    w_im[npix//2+1:npix] = -w_im[npix//2-1:0:-1]

    u_re = np.zeros([npix])
    u_re[1:npix//2] = random_2[1:npix//2] * np.sqrt(puk[1:npix//2] / 2)
    u_re[npix//2+1:npix] = u_re[npix//2-1:0:-1]

    u_im = np.zeros([npix])
    u_im[1:npix//2] = random_2[npix//2:npix-1] * np.sqrt(puk[1:npix//2] / 2)
    u_im[npix//2+1:npix] = -u_im[npix//2-1:0:-1]

    delta_k = (u_re + w_re) + 1j * (u_im + w_im)                                # delta_B = w(k,z) + u(k,z) 
    a_dot = np.sqrt((cosmo['h']*100)**2 * ( cosmo['omega_M'] * (1 + z) + cosmo['omega_L']/((1 + z)**2 )))
    v_k = a_dot * kval * beta_k * (-w_im + 1j * w_re)                           # v = i * a_dot * beta(k,z) * w(k,z)
    
    delta_x = therm_params['nu'] * np.fft.ifft(delta_k) * npix / boxsize ** 0.5
    v_x = np.fft.ifft(v_k) * npix / boxsize ** 0.5
    
    
    #############################################################################################################################
    nb = np.exp(np.real(delta_x))
    nH0= cosmo_functions.baryon_densities(**cosmo)[3] / Mpc ** 3
    nHe0= cosmo_functions.baryon_densities(**cosmo)[2] / Mpc ** 3
    nb0_z = (nH0 + nHe0) * (1 + z) ** 3
    norm_ln = nb0_z / np.mean(nb)
    nb = norm_ln * nb
    
    Tx = (10**therm_params['logT0']) * (nb / nb0_z) ** (therm_params['gamma'] - 1)
    b = np.sqrt(2.0 * kboltz * Tx / mprot) / km
    
    rec_alpha = 4.2e-13 * (Tx/1.e4) ** (-0.7)
    nprot = 4*(1. - cosmo['Y_He']) * nb / (4. - 3. * cosmo['Y_He'])
    ne = (4. - 2. * cosmo['Y_He']) * nb / (4. - 3. * cosmo['Y_He'])
    nHI = nprot * rec_alpha * ne / therm_params['Gamma']
    
    
    ##################################################
    ######   Voigt Profile Calculation Routines ######
    ##################################################
    size = round_up_to_even((1 + roll)*npix)
    zspectra = np.linspace(zmin - 0.5*roll*boxsize_z, zmax+0.5*roll*boxsize_z, num=size, endpoint=False)
    bspectra = np.r_[b[npix-(size-npix)//2:npix],b,b[0:(size-npix)//2]]         # make an array of size 2*npix
    vspectra = np.r_[v_x[npix-(size-npix)//2:npix],v_x,v_x[0:(size-npix)//2]]   # make an array of size 2*npix
    nHIspectra = np.r_[nHI[npix-(size-npix)//2:npix],nHI,nHI[0:(size-npix)//2]] # make an array of size 2*npix
    
    arr_ones = np.ones(size)
    onepz_b = np.outer((1 + zspectra), bspectra)
    x_ij = c_km_s * zspectra / onepz_b
    x_ij = x_ij - ((1 / onepz_b.T) * c_km_s * zspectra).T + vspectra / bspectra
    a_ij = np.outer(arr_ones, damp_gamma/bspectra)
    V_ij = voigt_wofz(a_ij, x_ij)
    F_j = nHIspectra / ((1 + zspectra) * bspectra)                              
    tauspectra = np.dot(V_ij, F_j)
    tau = tauspectra[(size-npix)//2:(size-npix)//2+npix]
    tau *= c_km_s * i_alpha * dx / np.sqrt(np.pi)                               # c * I * dx / sqrt(pi) 
    tau *= Mpc                                                                 
    field = np.real(np.c_[zval, delta_x, v_x, nb, Tx, nHI, tau])
    return field
 
    
 
############################# Thermal parameters ##############################
params = {}
params['xJ'] = 0.12                # in Mpc units
params['logT0'] = np.log10(10000)
params['gamma'] = 1.20
params['Gamma'] = 1.346e-12       
params['nu'] = 1.  
########################## Cosmological parameters ############################
cosmology = {}
cosmology['omega_b'] = 0.0456
cosmology['omega_M'] = 0.0456 + 0.227
cosmology['omega_L'] = 0.728
cosmology['omega_k'] = 0
cosmology['h'] = 0.704
cosmology['n'] = 0.996
cosmology['sigma_8'] = 0.809
cosmology['tau'] = 0.087
cosmology['z_reion'] = 10.4
cosmology['t_0'] = 13.75
cosmology['Y_He'] = 0.24
###############################################################################
#Linear matter power spectrum 
kpk_linear = np.loadtxt("pk_linear_camb.txt")
###############################################################################
a = lognormal_tau(cosmology, params, kpk_linear, 5, 0.05, 2.45, 2.55, 2048)
    
    
    
    
