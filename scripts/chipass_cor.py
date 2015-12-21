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

cmb_file = '../data/COM_CMB_IQU_sevem_256_deg.npz'
dust_file = '../data/COM_CompMap_dust-commander_0256_R2.00.fits'
sync_file = '../data/COM_CompMap_Synchrotron-commander_0256_R2.00.fits'
free_file = '../data/COM_CompMap_freefree-commander_0256_R2.00.fits'
radio_file = '../data/lambda_chipass_healpix_r10.fits'


f_cmb = np.load(cmb_file)
cmb_map =f_cmb['cmb']
cmb_mask = f_cmb['mask']

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
radio_map = hdu_radio[1].data.field('TEMPERATURE') * 1e-3 -2.725 #convert to  KCMB
counts = hdu_radio[1].data.field('SENSITIVITY')
hdu_radio.close()

sync_map = hp.reorder(sync_map,n2r=1)
dust_map = hp.reorder(dust_map,n2r=1)
free_EM = hp.reorder(free_EM,n2r=1)
free_T = hp.reorder(free_T,n2r=1)
radio_map = hp.reorder(radio_map,n2r=1)
counts = hp.reorder(counts,n2r=1)

##construct free-free intensity map
#
#gff = np.log( np.exp( 5.690 - np.sqrt(3.)/np.pi* np.log( freq * (free_T*1e-4)**(-1.5)) ) + np.e)
#tau = 0.05468 * (free_T)**(-1.5)*freq**(-2) * free_EM*gff
#free_map =  1e6*free_T*(1-np.exp(-tau))



radio_map[counts == 0] = hp.UNSEEN

radio_map = hp.smoothing(radio_map,fwhm=np.sqrt(60.0**2-14.4*82)*np.pi/(180.*60.))
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

##Remove foregrounds

gamma_sync = np.sum(radio_map*sync_map)/np.sum(sync_map**2) - np.sum(dust_map*sync_map)/np.sum(sync_map**2) *((np.sum(sync_map**2)*np.sum(radio_map*dust_map) - np.sum(radio_map*sync_map)*np.sum(sync_map*dust_map))/(np.sum(dust_map**2)*np.sum(sync_map**2) - np.sum(sync_map*dust_map)**2))

delta_dust = np.sum(radio_map*dust_map)/np.sum(dust_map**2) - np.sum(sync_map*dust_map)/np.sum(dust_map**2) *((np.sum(dust_map**2)*np.sum(radio_map*sync_map) - np.sum(radio_map*dust_map)*np.sum(sync_map*dust_map))/(np.sum(dust_map**2)*np.sum(sync_map**2) - np.sum(sync_map*dust_map)**2))


radio_fr = np.copy(radio_map - gamma_sync*sync_map - delta_dust * dust_map)

radio_fr = hp.ma(radio_fr)
radio_fr.mask=mask_bool


hp.mollview(radio_fr, norm='hist', unit='$K_{CMB}$')
plt.savefig('chipass_fr.png', format='png')
plt.close()

hp.mollview(radio_map, norm='hist', unit='$K_{CMB}$')
plt.savefig('chipass_raw.png', format='png')
plt.close()

cross_cls = hp.anafast(cmb_map,radio_fr)
cmb_cls = hp.anafast(cmb_map)

lmax = len(cross_cls)
l = np.arange(lmax)
ll = l*(l+1)/(2*np.pi)
beam = hp.gauss_beam(np.pi/180.,lmax-1)
pix = hp.pixwin(256)[:lmax]

wls = hp.anafast((~radio_fr.mask).astype(float))
#plt.plot(l,ll*wls)
#@plt.show()

#Mll = MLL.Mll(wls,l)

#Mll = np.array(Mll)

#np.savez('scalar_mixing_matrix.npz',mll=Mll)

Mll = np.load('scalar_mixing_matrix.npz')['mll']
#
#Mll = Mll.reshape(lmax,lmax)
U, S, V = np.linalg.svd((Mll*beam**2*pix**2).conj())

kll = np.einsum('ij,j,jk', V.T, 1./S, U.T)


cll = np.dot(kll, cross_cls)
cmb_cll = np.dot(kll,cmb_cls)


fsky = 1. - np.sum(mask_bool).astype(float)/len(mask_bool)
L = np.sqrt(4*np.pi*fsky)
dl_eff = 2*np.pi/L



fact = ll/fsky/(beam*pix)**2

#plt.plot(l,ll*cross,'k.')
#plt.plot(l,ll*cmb_cls,'r-')
#plt.show(block = False)

cross = cll.copy()
cmb_cls = cmb_cll.copy()

bcross= bin_llcl.bin_llcl(ll*cross,25)
bcmb  = bin_llcl.bin_llcl(ll*cmb_cls,25)

fig, ax = plt.subplots(1)

ax.plot(l,ll*cmb_cls*1e12, 'r-')
ax.errorbar(bcross['l_out'], bcross['llcl']*1e12, bcross['std_llcl']*1e12, fmt ='k.')
ax.set_xlabel('$\ell$')
ax.set_ylabel('$\\frac{\ell(\ell+1)}{2\pi} C_{\ell} [\mu K]^{2}$')

#ax.set_ylim([0,6e4])

fig.savefig('chipass_correlation_lin.png', fmt='png')

#ax.set_ylim([1,1e5])
ax.set_yscale('log')
fig.savefig('chipass_correlation_log.png', fmt='png')

likelihood(bcross['llcl'],bcross['std_llcl'],bcmb['llcl'],'chipass','fr')


