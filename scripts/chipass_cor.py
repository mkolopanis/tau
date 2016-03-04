import matplotlib
matplotlib.use('Agg')
import healpy as hp, numpy as np, matplotlib.pyplot as plt
from astropy.io import fits
import scalar_mixing_matrix as MLL
import bin_llcl
import ipdb
from scipy.interpolate import interp1d
def cov(x):
    #x -= x.mean()
    fact = x.shape[0]-1
    return np.dot(x.T,x)/fact

def likelihood(cross,dcross,theory,name,title):
        
        dcross=np.copy(dcross)

	a_scales=np.linspace(-50,50,100000)
	chi_array=[]
	for a in a_scales:
		chi_array.append(np.exp(-.5*np.sum( (cross - a*theory)**2/(dcross)**2)))
	chi_array /= np.max(chi_array)
	chi_sum = np.cumsum(chi_array)
	chi_sum /= chi_sum[-1]
	
	mean = a_scales[np.argmax(chi_array)]
	fig,ax1=plt.subplots(1,1)
	try:
		s1lo,s1hi = a_scales[chi_sum<0.1586][-1],a_scales[chi_sum>1-0.1586][0]
		s2lo,s2hi = a_scales[chi_sum<0.0227][-1],a_scales[chi_sum>1-0.0227][0]

		ax1.vlines(s1lo,0,1,linewidth=2,color='blue')
		ax1.vlines(s1hi,0,1,linewidth=2,color='blue')
		ax1.vlines(s2lo,0,1,linewidth=2,color='orange')
		ax1.vlines(s2hi,0,1,linewidth=2,color='orange')

                plt.xlim([s2lo - .5,s2hi + .5])
		f=open('Maximum_likelihood_optical_'+name+'_'+title+'.txt','w')
		f.write('Maximum Likelihood: {0:2.5f}%  for scale factor {1:.2f} \n'.format(float(chi_array[np.argmax(chi_array)]*100),mean))
		f.write('Posterior: Mean,\tsigma,\t(1siglo,1sighi),\t(2sighlo,2sighi)\n')
                f.write('Posterior: {0:.3f},\t{1:.3f} ,\t({2:.3f},{3:.3f})\t({4:.3f},{5:.3f})\n '.format(mean,np.mean([s1hi-mean,mean-s1lo]) ,s1lo,s1hi, s2lo,s2hi))
		f.write('Posterior SNR:\t {0:.3f}'.format(1./(np.mean( [s1hi-mean,mean-s1lo] )) ) )
		f.write('\n\n')
		f.close()

	except:
		print('Scale exceeded for possterior. \n Plotting anyway')
	ax1.plot(a_scales,chi_array,'k',linewidth=2)
	ax1.set_title('Posterior')
	#ax1.set_xlabel('Likelihood scalar')
	#ax1.set_ylabel('Likelihood of Correlation')
	#	
	fig.savefig('FR_simulation_likelihood_'+name+'_'+title+'.png',format='png')
	fig.savefig('FR_simulation_likelihood_'+name+'_'+title+'.eps',format='eps')
freq=1.42
bins = 25
#cmb_file = '../data/COM_CMB_IQU_sevem_256_deg.npz'
cmb_file ='../data/cmb_256.npy'
dust_file = '../data/COM_CompMap_dust-commander_0256_R2.00.fits'
sync_file = '../data/COM_CompMap_Synchrotron-commander_0256_R2.00.fits'
free_file = '../data/COM_CompMap_freefree-commander_0256_R2.00.fits'
radio_file = '../data/lambda_chipass_healpix_r10.fits'
wmap_file = '../data/wmap_mask_256.npy'
theory_cl_file= '../data/simul_scalCls.fits'

#f_cmb = np.load(cmb_file)
#cmb_map =f_cmb['cmb']
#cmb_mask = f_cmb['mask']
cmb_map = np.load(cmb_file)
cmb_mask = np.load(wmap_file)

hdu_dust = fits.open(dust_file)
dust_map = hdu_dust[1].data.field('I_ML') * 1e-6 -2.725 ##Convert K_RJ to K_CMB
hdu_dust.close()

hdu_free = fits.open(free_file)
free_EM = hdu_free[1].data.field('EM_ML')
free_T = hdu_free[1].data.field('TEMP_ML')
hdu_free.close()


hdu_sync = fits.open(sync_file)
sync_map = hdu_sync[1].data.field('I_ML') * 1e-6 -2.725  ##Convert K_RJ to K_CMB
hdu_sync.close()

hdu_radio = fits.open(radio_file)
radio_map = hdu_radio[1].data.field('TEMPERATURE') * 1e-3  -2.725#convert to  KCMB
counts = hdu_radio[1].data.field('SENSITIVITY')
hdu_radio.close()

sync_map = hp.reorder(sync_map,n2r=1)
dust_map = hp.reorder(dust_map,n2r=1)
free_EM = hp.reorder(free_EM,n2r=1)
free_T = hp.reorder(free_T,n2r=1)
radio_map = hp.reorder(radio_map,n2r=1)
counts = hp.reorder(counts,n2r=1)

cmb_mask = hp.ud_grade(cmb_mask,256)

##construct free-free intensity map
#
#gff = np.log( np.exp( 5.690 - np.sqrt(3.)/np.pi* np.log( freq * (free_T*1e-4)**(-1.5)) ) + np.e)
#tau = 0.05468 * (free_T)**(-1.5)*freq**(-2) * free_EM*gff
#free_map =  1e6*free_T*(1-np.exp(-tau))



radio_map[counts == 0] = hp.UNSEEN

#radio_map = hp.smoothing(radio_map,fwhm=np.sqrt(60.0**2-14.4*82)*np.pi/(180.*60.))
radio_map = hp.ud_grade(radio_map,256)

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

radio_map -= radio_map.mean()
cmb_map -= cmb_map.mean()
dust_map -= dust_map.mean()
sync_map -= sync_map.mean()

##High pass filter!

new_radio = radio_map  - hp.smoothing(radio_map,fwhm=np.sqrt((3*60.)**2-14.4**2)*np.pi/(180.*60.),verbose = False)

dust_map -= hp.smoothing(dust_map, fwhm=np.sqrt((3*60.)**2 - 60.**2)*np.pi/(180.*60),verbose = False)

sync_map -= hp.smoothing(sync_map, fwhm=np.sqrt((3*60.)**2 - 60.**2)*np.pi/(180.*60.), verbose = False)


##Remove foregrounds

gamma_sync = np.sum(new_radio*sync_map)/np.sum(sync_map**2) - np.sum(dust_map*sync_map)/np.sum(sync_map**2) *((np.sum(sync_map**2)*np.sum(new_radio*dust_map) - np.sum(new_radio*sync_map)*np.sum(sync_map*dust_map))/(np.sum(dust_map**2)*np.sum(sync_map**2) - np.sum(sync_map*dust_map)**2))

delta_dust = np.sum(new_radio*dust_map)/np.sum(dust_map**2) - np.sum(sync_map*dust_map)/np.sum(dust_map**2) *((np.sum(dust_map**2)*np.sum(new_radio*sync_map) - np.sum(new_radio*dust_map)*np.sum(sync_map*dust_map))/(np.sum(dust_map**2)*np.sum(sync_map**2) - np.sum(sync_map*dust_map)**2))

print 'Synchrotron sacle factor:', gamma_sync
print 'Dust scale factor:', delta_dust


radio_fr = new_radio - gamma_sync*sync_map - delta_dust * dust_map
radio_fr.mask= mask_bool

tmp_mask = (mask_bool).astype(float)
tmp_mask = hp.smoothing(tmp_mask,fwhm=1.*np.pi/(180.),verbose=False)
thresh = 5e-2
tmp_mask[tmp_mask < thresh] =0
tmp_mask[tmp_mask > thresh] =1
#tmp_mask1 = np.round(tmp_mask)
tmp_mask = tmp_mask.astype(bool)
new_mask = np.logical_or( tmp_mask, mask_bool)
#radio_fr = hp.ma(radio_fr)
radio_fr.mask = new_mask
radio_fr.fill_value = hp.UNSEEN


## re-flag previously unseen pixels
while(radio_fr.min() < -1e+30):
    radio_fr.mask[np.argmin(radio_fr).squeeze()] = True
while( radio_fr.max() / np.sqrt(np.mean(radio_fr**2)) > 4):
    radio_fr.mask[np.argmax(radio_fr).squeeze()] = True
while( radio_fr.min() / np.sqrt(np.mean(radio_fr**2)) > 4):
    radio_fr.mask[np.argmin(radio_fr).squeeze()] = True

new_mask = radio_fr.mask

radio_fr -= radio_fr.mean()

radio_map.mask = new_mask
cmb_map.mask  = new_mask

radio_map -= radio_map.mean()
cmb_map -= cmb_map.mean()
 
hp.mollview(radio_fr, norm='hist', unit='$K_{CMB}$')
plt.savefig('chipass_fr.png', format='png')
plt.close()

hp.mollview(radio_map, norm='hist', unit='$K_{CMB}$')
plt.savefig('chipass_raw.png', format='png')
plt.close()

cross_cls = hp.anafast(cmb_map,radio_fr)
radio_cls = hp.anafast(radio_fr)
cmb_cls = hp.anafast(cmb_map)



lmax = len(cross_cls)
beam_lmax = lmax
l = np.arange(2,beam_lmax)
ll = l*(l+1)/(2*np.pi)
beam_14 = hp.gauss_beam(14.4*np.pi/(180.*60.),beam_lmax-1)[2:]
beam_5 = hp.gauss_beam(5.*np.pi/(180.*60.),beam_lmax-1)[2:]
b14_180 = hp.gauss_beam(np.sqrt((3*60.)**2 - (14.4)**2)*np.pi/(180.*60.),beam_lmax-1)[2:]
b5_180 = hp.gauss_beam(np.sqrt((3*60.)**2 - (5.)**2)*np.pi/(180.*60.),beam_lmax-1)[2:]

beam_14 *= (1. - b14_180)
beam_5 *= (1. - b5_180)

pix = hp.pixwin(256)[2:beam_lmax]

theory_cls= hp.read_cl(theory_cl_file)
theory_cls=theory_cls[0][2:beam_lmax]
#theory_cls[:2]=1e-10

cross_cls = cross_cls[2:beam_lmax]
radio_cls = radio_cls[2:beam_lmax]
cmb_cls = cmb_cls[2:beam_lmax]

wls = hp.anafast((~radio_fr.mask).astype(float))[:beam_lmax]
fskyw2 = np.sum([(2*m+1)*wls[mi] if m > 0 else 0 for mi,m in enumerate(xrange(len(wls)))])/(4*np.pi)
wls = wls[2:]

fsky = 1. - np.sum(mask_bool).astype(float)/len(mask_bool)
L = np.sqrt(4*np.pi*fsky)
dl_eff = 2*np.pi/L



nbins = long((beam_lmax)/bins)
_lmax = nbins*bins -1
w=2*l+1

Pbl = np.tile( l*(l+1)/(2*np.pi*bins),(nbins,1))

Qlb = np.tile(2.*np.pi/(l*(l+1)).clip(1,np.Inf),(nbins,1))
Qlb = Qlb.T
Qlb[:2] = 0

q_mult = np.zeros_like(Qlb)
mult =np.zeros_like(Pbl)
for b in xrange(nbins):
    mult[b,bins*b :bins*b+bins -1 ] = 1. #add two to account for binning operator a la Hizon 2002
    q_mult[bins*b:bins*b+bins -1 ,b] = 1. #add two to account for binning operator a la Hizon 2002

Pbl *= mult
Qlb *= q_mult ## divide by .9 so np.dot(Pbl,Qlb) = identity


#norm = np.dot(Pbl,Qlb).diagonal()
#norm.shape = (1,norm.shape[0])

#Pbl /= np.sqrt(norm)
#Qlb /= np.sqrt(norm)

#Pbl = Pbl[:, 2:beam_lmax]
#Qlb = Pbl[2:beam_lmax, :]

l_out = bin_llcl.bin_llcl(ll,bins)['l_out']
#bcross_cls= bin_llcl.bin_llcl(ll*cross_cls,bins)
#bcmb_cls = bin_llcl.bin_llcl(ll*cmb_cls,bins)
#bwls = bin_llcl.bin_llcl(ll*wls[2:],bins)



Mll = MLL.Mll(wls,l)
#Mll = np.array(Mll)
np.savez('mll_chipass.npz',mll=Mll)

#Mll = np.load('mll_chipass.npz')['mll']
#Mll = Mll[2:beam_lmax,2:beam_lmax]
#Mll = Mll.reshape(lmax,lmax)

#compute TOD transfer function.. Maybe
#N_cmb = 100
#
#cl_mc=[]
#print 'Creaing transform function'
##for n in xrange(N_cmb):
##    temp_cmb = hp.synfast(theory_cls,nside=256,fwhm=0,verbose=False,pixwin=True)
##
##    temp_cmb = hp.smoothing(temp_cmb,fwhm=5.*np.pi/(180.*60.),verbose=False)
##
##    temp_cmb = hp.ma(temp_cmb)
##    temp_cmb.mask = mask_bool
##
##
##    cl_mc.append(hp.anafast(temp_cmb)[2:beam_lmax])
##
##cl_avg=np.mean(cl_mc,axis=0)
#
#
##bnoise=bin_llcl.bin_llcl(ll*noise_mc,bins)
##binned_t = bin_llcl.bin_llcl(ll*theory_cls,bins)
#binned_t={}
#binned_t['llcl'] = np.einsum('ij,j', Pbl, theory_cls)

#S_cl_avg = np.convolve(cl_avg, np.ones(50)/50.,mode='same')

F0_cmb = np.array([ 1-2./np.pi*np.arcsin(25./n) if n >= 25 else 0 for n in xrange(3,beam_lmax+1)])



kbb_cross = np.einsum('ij,jk,kl', Pbl, Mll *beam_14*beam_5*pix**2*F0_cmb,Qlb)
kbb_radio = np.einsum('ij,jk,kl', Pbl, Mll *beam_14**2*pix**2*F0_cmb,Qlb)
kbb_cmb = np.einsum('ij,jk,kl',  Pbl, Mll *beam_5**2*pix**2*F0_cmb,Qlb)


U, S, V = np.linalg.svd(kbb_cross)
_kbb_cross = np.einsum('ij,j,jk', V.T, 1./S, U.T)


U1, S1, V1 = np.linalg.svd(kbb_cmb)
_kbb_cmb = np.einsum('ij,j,jk', V1.T, 1./S1, U1.T)

U2, S2, V2 = np.linalg.svd(kbb_radio)
_kbb_radio=  np.einsum('ij,j,jk', V2.T, 1./S2, U2.T)

dll = {}
cmb_dll = {}
noise_dll = {}
radio_dll = {}
#dll['llcl'] = np.einsum('ij,j', _kbb_cross, bcross_cls['llcl'])
dll['llcl'] = np.einsum('ij,jk,k', _kbb_cross, Pbl,cross_cls)
#dll['std_llcl'] = np.einsum('ij,j',_kbb_cross, bcross_cls['std_llcl'])
#cmb_dll['llcl'] = np.einsum('ij,j',_kbb_cmb,bcmb_cls['llcl'])
cmb_dll['llcl'] = np.einsum('ij,jk,k',_kbb_cmb, Pbl, cmb_cls)
radio_dll['llcl'] = np.einsum('ij,jk,k', _kbb_radio,Pbl,radio_cls)
#cmb_dll['std_llcl'] = np.einsum('ij,j',_kbb_cmb,bcmb_cls['std_llcl'])
#noise_dll['llcl'] = np.einsum('ij,kj', _kbb_cross,bnoise['llcl'])

##interpolate to find new base cls

#cl_interp= interp1d(l_out,dll['llcl'],bounds_error=0,kind='linear')
#
#new_cls = cl_interp(xrange(lmax))
#new_cls = np.ma.masked_invalid(new_cls)
#new_cls.fill_value=0.0
#
##now need to find error bars
#noise_mc=[]
#noise_const = 400e-6
#print 'Creating Error bars with {0} cmb realization'.format(N_cmb)
#for n in xrange(N_cmb):
#    temp_cmb = hp.synfast(new_cls.filled(),nside=256,fwhm=0,verbose=False,pixwin=True)
#    temp_noise = np.copy(temp_cmb) + noise_const*np.random.normal(size=len(temp_cmb))
#    temp_noise = hp.smoothing(temp_noise, fwhm=14.4*np.pi/(180.*60.),verbose=False)
#    temp_noise = hp.ma(temp_noise)
#    temp_noise.mask = mask_bool
#    noise_mc.append(hp.anafast(temp_noise,temp_cmb)[2:beam_lmax])
#
#noise_dll['llcl'] = np.einsum('ij,jk,lk', _kbb_cross,Pbl, noise_mc)
#
#Cov = cov( dll['llcl'] - noise_dll['llcl'].T)
#
#delta = np.sqrt(Cov.diagonal())

F0_bin = bin_llcl.bin_llcl(F0_cmb,bins,uniform=True)['llcl']

delta= np.sqrt(2./((2*l_out+1)*np.sqrt(dl_eff**2+bins**2)*fskyw2*F0_bin)* (cmb_dll['llcl']**2 + abs(cmb_dll['llcl']*radio_dll['llcl'])/2.) )

fig, ax = plt.subplots(1)
good_l = np.logical_and(l_out>25, l_out <= 500)

ax.plot(l_out[good_l],cmb_dll['llcl'][good_l]*1e12, 'r-')
ax.errorbar(l_out[good_l], dll['llcl'][good_l]*1e12, delta[good_l]*1e12, fmt ='k.')
ax.set_xlabel('$\ell$')
ax.set_ylabel('$\\frac{\ell(\ell+1)}{2\pi} C_{\ell} [\mu K]^{2}$')

#ax.set_ylim([0,6e4])

fig.savefig('chipass_correlation_lin.png', fmt='png')

#ax.set_ylim([1,1e5])
ax.set_yscale('log')
fig.savefig('chipass_correlation_log.png', fmt='png')

fit_l = np.logical_and(l_out>25, l_out < 500)

likelihood(dll['llcl'][good_l],delta[good_l],cmb_dll['llcl'][good_l],'chipass','fr')


