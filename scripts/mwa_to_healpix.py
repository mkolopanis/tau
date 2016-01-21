import numpy as np
import healpy as hp
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as units

def regioncoords(header):

    if header is None:
        print 'Must supply header type'
        return

    pix1,pix2 = header['crpix1'], header['crpix2']
    dx1, dx2 = header['cdelt1'], header['cdelt2']
    ra,dec = header['crval1'], header['crval2']

    npix1, npix2 = header['naxis1'], header['naxis2']


    decs = np.array( [ dec + dx2*(pix2 - j) for j  in xrange( int(npix2) )] )
    coords = np.array( [[ np.mod( ra  + dx1*( pix1 - j )/np.cos(dec*np.pi/180.),360),dec] for j in xrange( int(npix1) ) for dec in decs ] )

    return coords

nside = 256
mwa_file = '../data/mwa_1hr_residual.fits'

hdu_mwa = fits.open(mwa_file)
mwa_header = hdu_mwa[0].header
mwa_img = hdu_mwa[0].data.squeeze()
hdu_mwa.close()



coords = regioncoords(mwa_header)
coords_sky = SkyCoord(ra= coords[:,0], dec= coords[:,1],unit=units.degree,frame='fk5')

phi = coords_sky.galactic.l.deg*np.pi/180.
theta = (90. - coords_sky.galactic.b.deg)*np.pi/180.

mwa_pix = hp.ang2pix(nside,theta,phi)

mwa_map = np.zeros(hp.nside2npix(nside))
mwa_count = np.zeros(hp.nside2npix(nside))

x,y = np.meshgrid( xrange(mwa_header['naxis1']) , xrange(mwa_header['naxis2']))


for pix,i,j in zip(mwa_pix,x.flat,y.flat):
    mwa_map[pix] += mwa_img[i,j]
    mwa_count[pix] += 1.

mwa_map[mwa_count != 0] /= mwa_count[mwa_count != 0]


np.savez('mwa_1hr_residual_256.npz', image= mwa_map , counts=mwa_count)

