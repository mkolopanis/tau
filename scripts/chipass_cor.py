import matplotlib
#matplotlib.use('Agg')
import healpy as hp, numpy as np, matplotlib.pyplot as plt
from astropy.io import fits
import scalar_mixing_matrix as MLL
import bin_llcl

def likelihood(cross,dcross,theory,name,title):
        
        dcross=np.copy(dcross)

	a_scales=np.linspace(-1000,1000,100000)
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
bins = 10
#cmb_file = '../data/COM_CMB_IQU_sevem_256_deg.npz'
cmb_file ='../data/cmb_256.npy'
dust_file = '../data/COM_CompMap_dust-commander_0256_R2.00.fits'
sync_file = '../data/COM_CompMap_Synchrotron-commander_0256_R2.00.fits'
free_file = '../data/COM_CompMap_freefree-commander_0256_R2.00.fits'
radio_file = '../data/lambda_chipass_healpix_r10.fits'
wmap_file = '../data/wmap_mask_256.npy'

#f_cmb = np.load(cmb_file)
#cmb_map =f_cmb['cmb']
#cmb_mask = f_cmb['mask']
cmb_map = np.load(cmb_file)
cmb_mask = np.load(wmap_file)

hdu_dust = fits.open(dust_file)
dust_map = hdu_dust[1].data.field('I_ML') * 1e-6 ##Convert K_RJ to K_CMB
hdu_dust.close()

hdu_free = fits.open(free_file)
free_EM = hdu_free[1].data.field('EM_ML')
free_T = hdu_free[1].data.field('TEMP_ML')
hdu_free.close()


hdu_sync = fits.open(sync_file)
sync_map = hdu_sync[1].data.field('I_ML') * 1e-6  ##Convert K_RJ to K_CMB
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



##Remove foregrounds

gamma_sync = np.sum(radio_map*sync_map)/np.sum(sync_map**2) - np.sum(dust_map*sync_map)/np.sum(sync_map**2) *((np.sum(sync_map**2)*np.sum(radio_map*dust_map) - np.sum(radio_map*sync_map)*np.sum(sync_map*dust_map))/(np.sum(dust_map**2)*np.sum(sync_map**2) - np.sum(sync_map*dust_map)**2))

delta_dust = np.sum(radio_map*dust_map)/np.sum(dust_map**2) - np.sum(sync_map*dust_map)/np.sum(dust_map**2) *((np.sum(dust_map**2)*np.sum(radio_map*sync_map) - np.sum(radio_map*dust_map)*np.sum(sync_map*dust_map))/(np.sum(dust_map**2)*np.sum(sync_map**2) - np.sum(sync_map*dust_map)**2))

print 'Synchrotron sacle factor:', gamma_sync
print 'Dust scale factor:', delta_dust


radio_fr = np.copy(radio_map.data - gamma_sync*sync_map.data - delta_dust * dust_map.data)

radio_fr = hp.ma(radio_fr)
radio_fr.mask=mask_bool
radio_fr -=radio_fr.mean()

hp.mollview(radio_fr, norm='hist', unit='$K_{CMB}$')
plt.savefig('chipass_fr.png', format='png')
plt.close()

hp.mollview(radio_map, norm='hist', unit='$K_{CMB}$')
plt.savefig('chipass_raw.png', format='png')
plt.close()

cross_cls = hp.anafast(cmb_map,radio_fr-radio_fr.mean())
cmb_cls = hp.anafast(cmb_map)
wls = hp.anafast((~mask_bool).astype(float))

lmax = len(cross_cls)
l = np.arange(lmax)
ll = l*(l+1)/(2*np.pi)
beam_14 = hp.gauss_beam(14.4*np.pi/(180.*60.),lmax-1)
beam_5 = hp.gauss_beam(14.4*np.pi/(180.*60.),lmax-1)
pix = hp.pixwin(256)[:lmax]

wls = hp.anafast((~radio_fr.mask).astype(float))


nbins = long((lmax-2)/bins)
_lmax = nbins*bins -1 +2
m=np.arange(lmax+1)
w=2*l+1

Pbl = np.tile( l*(l+1)/(2*np.pi*25),(nbins,1))

mult =np.zeros_like(Pbl)
for b in xrange(nbins):
    mult[b,bins*b +2:bins*b+bins -1 +2] = 1. #add two to account for binning operator a la Hizon 2002


Pbl *= mult

Qlb = np.tile(2*np.pi/(l*(l+1)).clip(1,np.Inf),(nbins,1)).T
Qlb[0] = 0.

Qlb *= mult.T
#plt.plot(l,ll*wls)
#@plt.show()

l_out = bin_llcl.bin_llcl(ll,bins)['l_out']
bcross_cls= bin_llcl.bin_llcl(ll*cross_cls,bins)
bcmb_cls = bin_llcl.bin_llcl(ll*cmb_cls,bins)
bwls = bin_llcl.bin_llcl(ll*wls,bins)


_lmax = bins*len(bcross_cls)
#l = np.arange(lmax)
#ll= l*(l+1)/(2*np.pi)
#_ll = 2*np.pi/(l*(l+1)).clip(1,np.Inf)

#ubcross = _ll * np.repeat(bcross_cls,bins)
#ubcmb_cls = _ll * np.repeat(bcmb_cls,bins)
#ubwls = _ll * np.repeat(bwls,bins)

#Mll = MLL.Mll(wls,l)

#Mll = np.array(Mll)

#np.savez('scalar_mixing_matrix.npz',mll=Mll)

Mll = np.load('scalar_mixing_matrix.npz')['mll']
#
#Mll = Mll.reshape(lmax,lmax)

kbb_cross = np.dot(Pbl, np.dot(Mll *beam_14*beam_5*pix**2,Qlb))
kbb_cmb = np.dot(Pbl, np.dot(Mll *beam_5**2*pix**2,Qlb))


U, S, V = np.linalg.svd(kbb_cross)

_kbb_cross = np.einsum('ij,j,jk', V.T, 1./S, U.T)[:lmax,:lmax]
U, S, V = np.linalg.svd(kbb_cmb)

_kbb_cmb = np.einsum('ij,j,jk', V.T, 1./S, U.T)[:lmax,:lmax]

dll= {}
cmb_dll= {}
for key in bcross_cls.keys():
    dll[key] = np.dot(_kbb_cross, bcross_cls[key])
    cmb_dll[key] =np.dot(_kbb_cmb,bcmb_cls[key])

#cll = np.dot(kll, ubcross)
#cmb_cll = np.dot(kll,ubcmb_cls)


fsky = 1. - np.sum(mask_bool).astype(float)/len(mask_bool)
L = np.sqrt(4*np.pi*fsky)
dl_eff = 2*np.pi/L




#plt.plot(l,ll*cross,'k.')
#plt.plot(l,ll*cmb_cls,'r-')
#plt.show(block = False)

#cross = cll.copy()
#cmb_cls = cmb_cll.copy()
#
#bcross= bin_llcl.bin_llcl(ll*cross,bins)
#bcmb  = bin_llcl.bin_llcl(ll*cmb_cls,bins)

fig, ax = plt.subplots(1)

ax.plot(l_out,cmb_dll['llcl']*1e12, 'r-')
ax.errorbar(l_out, dll['llcl']*1e12, dll['std_llcl']*1e12, fmt ='k.')
ax.set_xlabel('$\ell$')
ax.set_ylabel('$\\frac{\ell(\ell+1)}{2\pi} C_{\ell} [\mu K]^{2}$')

#ax.set_ylim([0,6e4])

fig.savefig('chipass_correlation_lin.png', fmt='png')

#ax.set_ylim([1,1e5])
ax.set_yscale('log')
fig.savefig('chipass_correlation_log.png', fmt='png')

likelihood(dll['llcl'],dll['std_llcl'],cmb_dll['llcl'],'chipass','fr')


