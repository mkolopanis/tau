import matplotlib
#matplotlib.use('Agg')
import healpy as hp, numpy as np, matplotlib.pyplot as plt
from astropy.io import fits
import bin_llcl


cmb_file = '../data/COM_CMB_IQU_seven_256_deg.npz'
dust_file = '../data/COM_CompMap_dust-commander_0256_R2.00.fits'
sync_file = '../data/COM_CompMap_Synchrotron-commander_0256_R2.00.fits'
radio_file = '../data/lambda_chipass_healpix_r10.fits'


f_cmb = np.load(cmb_file)
cmb_map =f_cmb['cmb']
cmb_mask = f_cmb['mask']

hdu_dust = fits.open(dust_file)
dust_map = hdu_dust[1].data.field('I_ML') * 1e-6 ##Convert K_RJ to K_CMB
hdu_dust.close()


hdu_sync = fits.open(sync_file)
sync_map = hdu_sync[1].data.field('I_ML') * 1e-6 ##Convert K_RJ to K_CMB
hdu_sync.close()

hdu_radio = fits.open(radio_file)
radio_map = hdu_radio[1].data.field('TEMPERAUTRE') * 1e-3 #conver to K
counts = hdu_radio[1].data.field('SENSITIVITY')
hdu_radio.close()

sync_map = hp.reoder(sync_map,n2r=1)
dust_map = hp.reoder(dust_map,n2r=1)
radio_map = hp.reorder(radio_map,n2r=1)
counts = hp.reorder(counts,n2r=1)

radio_map = hp.smoothing(radio_map,fwhm=np.pi/180.)

mask = np.logical_and(mask,counts)

mask_bool = ~mask.astype(bool)

