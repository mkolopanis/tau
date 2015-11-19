import matplotlib
matplotlib.use('Agg')
import healpy as hp, numpy as np, matplotlib.pyplot as plt
from astropy.io import fits
import bin_llcl


cmb_file = '../data/COM_CMB_IQU_sevem_256_deg.npz'
dust_file = '../data/COM_CompMap_dust-commander_0256_R2.00.fits'
sync_file = '../data/COM_CompMap_Synchrotron-commander_0256_R2.00.fits'
radio_file = '../data/lambda_chipass_healpix_r10.fits'


f_cmb = np.load(cmb_file)
cmb_map =f_cmb['cmb']
cmb_mask = f_cmb['mask']

hdu_dust = fits.open(dust_file)
dust_map = hdu_dust[1].data.field('I_ML') * 1e-6 -2.73 ##Convert K_RJ to K_CMB
hdu_dust.close()


hdu_sync = fits.open(sync_file)
sync_map = hdu_sync[1].data.field('I_ML') * 1e-6 -2.73 ##Convert K_RJ to K_CMB
hdu_sync.close()

hdu_radio = fits.open(radio_file)
radio_map = hdu_radio[1].data.field('TEMPERATURE') * 1e-3 -2.73 #conver to  KCMB
counts = hdu_radio[1].data.field('SENSITIVITY')
hdu_radio.close()

sync_map = hp.reorder(sync_map,n2r=1)
dust_map = hp.reorder(dust_map,n2r=1)
radio_map = hp.reorder(radio_map,n2r=1)
counts = hp.reorder(counts,n2r=1)

radio_map[counts == 0] = 0

radio_map = hp.ud_grade(radio_map,256)
radio_map = hp.smoothing(radio_map,fwhm=np.pi/180.)

counts = hp.ud_grade(counts,256)

mask = np.logical_and(cmb_mask,counts)

mask_bool = ~mask.astype(bool)

radio_back=np.copy(radio_map)
radio_map = hp.ma(radio_map)
cmb_map = hp.ma(cmb_map)
dust_map = hp.ma(dust_map)
sync_map = hp.ma(sync_map)

radio_map.mask = mask_bool
cmb_map.mask = mask_bool
dust_map.mask = mask_bool
sync_map.mask = mask_bool

##Remove foregrounds

gamma_sync = np.sum(radio_map*sync_map)/np.sum(sync_map**2) - np.sum(dust_map*sync_map)/np.sum(sync_map**2) *((np.sum(sync_map**2)*np.sum(radio_map*dust_map) - np.sum(radio_map*sync_map)*np.sum(sync_map*dust_map))/(np.sum(dust_map**2)*np.sum(sync_map**2) - np.sum(sync_map*dust_map)**2))

delta_dust = np.sum(radio_map*dust_map)/np.sum(dust_map**2) - np.sum(sync_map*dust_map)/np.sum(dust_map**2) *((np.sum(dust_map**2)*np.sum(radio_map*sync_map) - np.sum(radio_map*dust_map)*np.sum(sync_map*dust_map))/(np.sum(dust_map**2)*np.sum(sync_map**2) - np.sum(sync_map*dust_map)**2))


radio_fr = np.copy(radio_map - gamma_sync*sync_map - delta_dust * dust_map)

radio_fr = hp.ma(radio_fr)
radio_fr.mask=mask_bool

cross = hp.anafast(cmb_map,radio_fr)

fsky = 1. - np.sum(mask_bool).astype(float)/len(mask_bool)
L = np.sqrt(4*np.pi*fsky)
dl_eff = 2*np.pi/L

lmax = len(cross)
l = np.arange(lmax)
ll = l*(l+1)/(2*np.pi)

beam = hp.gauss_beam(np.pi/180.,lmax-1)
pix = hp.pixwin(256)[:lmax]

fact = ll/fsky/(beam*pix)**2


