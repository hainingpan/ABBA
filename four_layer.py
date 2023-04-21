from pathlib import Path
import pandas as pd
import numpy as np
path_A = Path('lattice_exp.csv')
path_B = Path('/mnt/d/Cornell/ABBA/lattice_exp.csv')
path= path_A if path_A.exists() else path_B
data_lattice=pd.read_csv(path,delimiter=' ',)
data_lattice=data_lattice.set_index('Angstrom')
class Params:
    def __init__(self,a=data_lattice['WSe2']['a'],m=1.2, theta=3, Tm=157,Tp_0=389,Tp_1=-1,V1=-89.6,V2_0=-83.3,V2_1=-1,phi2=-0.758,Ez=0,Nmax=2,n=15,B=0):
        self.m=m
        self.a=a
        self.theta=deg2rad(theta)   
        self.Tm=Tm
        self.Tp_0=Tp_0
        self.Tp_1=Tp_1
        self.V1=V1
        self.V2_0=V2_0
        self.V2_1=V2_1
        self.phi2=phi2
        self.Ez=Ez
        self.Nmax=Nmax
        self.hbar2=(1.05457182e-34)**2/(9.1e-31)/(1e-10)**2/1.602e-19*1e3
        self.aM=self.a/(self.theta)  # amstrong
        # conventional def
        self.aM1=self.aM*np.array([np.cos(deg2rad(-30)),np.sin(deg2rad(-30))])
        self.aM2=self.aM*np.array([np.cos(deg2rad(30)),np.sin(deg2rad(30))])

        self.bM=4*np.pi/(np.sqrt(3)*self.aM)    # 1/amstrong
        self.g=[np.array([np.cos(deg2rad(x)),np.sin(deg2rad(x))])*self.bM for x in np.arange(6)*60]
        self.bM1=self.g[5]
        self.bM2=self.g[1]
        self.bM3=self.g[3]

        self.g_idx=np.array(self.g)@np.linalg.inv(np.array([self.bM1,self.bM2]))

        self.bM1_idx=self.g_idx[5]
        self.bM2_idx=self.g_idx[1]
        self.bM3_idx=self.g_idx[3]
        self.G=np.array([self.bM1,self.bM2,self.bM3])

        self.n=n

        self.neighbor_index=generate_shell(self.Nmax)
        self.h1index=self.neighbor_index[:,0]
        self.h2index=self.neighbor_index[:,1]

        self.G_idx=np.array([self.bM1_idx,self.bM2_idx,self.bM3_idx])
        self.G_m_idx=-self.G_idx
        h1matX,h1matY,h2matX,h2matY=self._get_h()
        self.Tmmat=self._T(h1matX-h1matY,h2matX-h2matY,layer='-')
        self.TmTmat=self.Tmmat.T.conj() # TRS ensures the hermicity
        self.Tpmat=self._T(h1matX-h1matY,h2matX-h2matY,layer='+')
        self.TpTmat=self.Tpmat.T.conj() # TRS ensures the hermicity
        self.Vb1mat=self._V(h1matX-h1matY,h2matX-h2matY,layer=1)
        self.Vb2mat=self._V(h1matX-h1matY,h2matX-h2matY,layer=2)
        self.Vb3mat=self._V(h1matX-h1matY,h2matX-h2matY,layer=3)
        self.Vb4mat=self._V(h1matX-h1matY,h2matX-h2matY,layer=4)
        self.high_symm=self._generate_high_symm()
        mu_B,g=5.78e-5,10
        self.Vz=g*mu_B*B*1e3

    def _generate_high_symm(self):
        K_=(self.bM1+2*self.bM2)/3
        Kp_=(-self.bM1+self.bM2)/3
        M_=self.bM2/2
        G_=0*self.bM1

        G_M_K_G=[G_,M_,K_,G_]
        K_M_Kp_G=[K_,M_,Kp_,G_]

        G_M_K_G_kx,G_M_K_G_ky=_interpolate_path(G_M_K_G,self.n)
        K_M_Kp_G_kx,K_M_Kp_G_ky=_interpolate_path(K_M_Kp_G,self.n)
        
        G_M_K_G_dist=np.r_[0,np.cumsum(np.sqrt(np.diff(G_M_K_G_kx)**2+np.diff(G_M_K_G_ky)**2))]

        K_M_Kp_G_dist=np.r_[0,np.cumsum(np.sqrt(np.diff(K_M_Kp_G_kx)**2+np.diff(K_M_Kp_G_ky)**2))]

        G_M_K_G_name={G_M_K_G_dist[idx*self.n]:n for idx,n in enumerate([r'$\Gamma$',r'M',r'$\kappa$',r'$\Gamma$'])}

        K_M_Kp_G_name={K_M_Kp_G_dist[idx*self.n]:n for idx,n in enumerate([r'$\kappa$',r'M',r'$\kappa^\prime$',r'$\Gamma$'])}

        nshell=int(self.n)
        u_index=generate_shell(nshell)/nshell
        k_hex=u_index@np.array([self.bM1,self.bM2])/np.sqrt(3)@_rotate(deg2rad(30))

        ux,uy=np.mgrid[0:self.n,0:self.n]
        u_index=np.c_[ux.flatten()/self.n,uy.flatten()/self.n]
        k_diamond=u_index@np.array([self.bM1,self.bM2])
        
        return {'kappa':K_,'M':M_,'Gamma':G_,'kappa_p':Kp_,
        'G_M_K_G':(G_M_K_G_kx,G_M_K_G_ky),'G_M_K_G_dist':G_M_K_G_dist,'G_M_K_G_name':G_M_K_G_name,
        'K_M_Kp_G':(K_M_Kp_G_kx,K_M_Kp_G_ky),'K_M_Kp_G_dist':K_M_Kp_G_dist,'K_M_Kp_G_name':K_M_Kp_G_name,
        'hex':k_hex,'diamond':k_diamond,
        }

    def _get_h(self):
        h1matX,h1matY=np.meshgrid(self.h1index,self.h1index,indexing='ij')
        h2matX,h2matY=np.meshgrid(self.h2index,self.h2index,indexing='ij')
        return h1matX,h1matY,h2matX,h2matY

    def _T(self,h1,h2,layer):
        '''
        layer: + Tp
        layer: - Tm
        '''

        # h1,h2=h1*self.t,h2*self.t
        if layer=='-':
            w0_sum=self.Tm*(np.isclose(h1,0)*np.isclose(h2,0))
            return w0_sum
        else:
            w0_sum=self.Tp_0*(np.isclose(h1,0)*np.isclose(h2,0))
            w1_sum=self.Tp_1*(np.sum([np.isclose(h1,idx[0])*np.isclose(h2,idx[1]) for idx in np.vstack([self.G_idx,self.G_m_idx])],axis=0))
            return w0_sum+w1_sum

    def _V(self,h1,h2,layer):
        '''
        l=+1: bottom layer
        l=-1: top layer
        t=+1/-1: TRS
        '''
        # h1,h2=h1*self.t,h2*self.t
        phi=self.phi2
        l=1 if layer<=2 else -1
        if layer==1 or layer==4:
            return self.V1*(np.isclose(h1,0)*np.isclose(h2,0))
        else:
            return self.V2_0*(np.isclose(h1,0)*np.isclose(h2,0)) +         self.V2_1*(
            np.exp(-1j*phi*l)*np.sum([np.isclose(h1,idx[0])*np.isclose(h2,idx[1]) for idx in self.G_idx],axis=0)
        +np.exp(1j*phi*l)*np.sum([np.isclose(h1,idx[0])*np.isclose(h2,idx[1]) for idx in self.G_m_idx],axis=0)
        )

    def energy_bonding(self,kx,ky):
        klist=np.c_[kx,ky]+self.h1index[:,np.newaxis]*self.bM1+self.h2index[:,np.newaxis]*self.bM2
        H=-self.hbar2/(2*self.m)*np.diag(np.sum(klist*klist,axis=1))
        return H
        
    def energy_single(self,kx,ky):
        # kblist=np.c_[kx,ky]+self.h1index[:,np.newaxis]*self.bM1+self.h2index[:,np.newaxis]*self.bM2
        # ktlist=kblist
        # kblist=kblist*self.t
        # ktlist=ktlist*self.t
        # kblist[:,0]=self.C2y*kblist[:,0]
        # ktlist[:,0]=self.C2y*ktlist[:,0]
        T=self.energy_bonding(kx,ky)

        H11=T+self.Ez*3/2*np.eye(T.shape[0])+self.Vb1mat
        H22=T+self.Ez*1/2*np.eye(T.shape[0])+self.Vb2mat
        H33=T-self.Ez*1/2*np.eye(T.shape[0])+self.Vb3mat
        H44=T-self.Ez*3/2*np.eye(T.shape[0])+self.Vb4mat
        H12=self.Tmmat
        H21=self.TmTmat
        H23=self.Tpmat
        H32=self.TpTmat
        H34=self.TmTmat
        H43=self.TmTmat
        zeros=np.zeros((T.shape[0],T.shape[0]))
        H=np.block([[H11,H12,zeros,zeros],
                    [H21,H22,H23,zeros],
                    [zeros,H32,H33,H34],
                    [zeros,zeros,H43,H44]])
        H=np.kron(np.eye(2),H)+self.Vz*np.kron(np.array([[0,1],[1,0]]),np.eye(H.shape[0]))

        val,vec=np.linalg.eigh(H)
        vec=self._correct_T(vec)
        return val[::-1],vec[:,::-1],H
    def _correct_T(self,vec,phase=None):
        '''
        Correct U(1) for TRS, because Gamma is always at the center
        '''
        if phase is None:
            phase=np.angle(vec[self.h1index.shape[0]//2,:])
        return vec@np.diag(np.exp(-1j*phase))
    def plot_dispersion(self,ax=None,k=10,line='G_M_K_G',color='k'):
        '''
        take 10 maximal band
        '''
        assert line in {'G_M_K_G','K_M_Kp_G'}, 'line should be "K_M_Kp_G" or "G_M_K_G"'
        energy_list=[]
        for kx,ky in zip(*self.high_symm[line]):
            val,vec,_=self.energy_single(kx, ky)
            energy_list.append(val)
        
        energy_list=np.array(energy_list)
        if ax is not False:
            if ax is None:
                fig,ax=plt.subplots(figsize=(4,4/self.aM1[0]*self.aM2[1]))
            ax.plot(self.high_symm[line+'_dist'],energy_list[:,:k],color=color)

            xticks,xticklabels=[],[]
            for pos,name in self.high_symm[line+'_name'].items():
                ax.axvline(x=pos,ls='dashed',color='k',lw=1)
                xticks.append(pos)
                xticklabels.append(name)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)

            ax.set_ylabel(r'$E$ (meV)')
            ax.set_xlim(self.high_symm[line+'_dist'][[0,-1]])
            # ax.set_ylim(energy_list[:,k].min(),energy_list[:,0].max()) 

        return energy_list
    #plot_bandstructure
    #LDOS_r
    #plot_LDOS


    def u(self,vec,rx,ry):
        expo=np.exp(1j*(np.c_[self.h1index,self.h2index]@np.array([self.bM1,self.bM2]))@np.c_[rx,ry].T)
        vec1=vec[:vec.shape[0]//4]
        vec2=vec[vec.shape[0]//4:vec.shape[0]//2]
        vec3=vec[vec.shape[0]//2:3*vec.shape[0]//4]
        vec4=vec[3*vec.shape[0]//4:]
        u1=vec1@expo
        u2=vec2@expo
        u3=vec3@expo
        u4=vec4@expo
        return u1,u2,u3,u4

    def psi(self,vec,rx,ry,kx,ky):
        expo=np.exp(1j*(np.c_[self.h1index,self.h2index]@np.array([self.bM1,self.bM2])+np.array([kx,ky]))@np.c_[rx,ry].T)
        vec1=vec[:vec.shape[0]//4]
        vec2=vec[vec.shape[0]//4:vec.shape[0]//2]
        vec3=vec[vec.shape[0]//2:3*vec.shape[0]//4]
        vec4=vec[3*vec.shape[0]//4:]
        psi1=vec1@expo
        psi2=vec2@expo
        psi3=vec3@expo
        psi4=vec4@expo
        return psi1,psi2,psi3,psi4

    def plot_lattice(self,nshell,ax=None):
        line_1=np.array([[0,0],self.aM1])/self.aM
        line_2=np.array([[0,0],self.aM2])/self.aM
        line_3=np.array([[0,0],-self.aM1+self.aM2])/self.aM
        if ax is None:
            fig,ax=plt.subplots()
        neighbor_list=generate_shell(nshell)
        for h1,h2 in neighbor_list:
            for line in [line_1,line_2,line_3]:
                line=line+h1*self.aM1/self.aM+h2*self.aM2/self.aM
                ax.plot(*(line.T),lw=1,color='k')

    def plot_psi(self,kx,ky,state,ax=None,resolution=50):
        rx_list=np.linspace(-self.aM1[0],self.aM1[0],resolution+1)
        ry_list=np.linspace(-2*self.aM2[1],2*self.aM2[1],resolution+1)
        rx_mat,ry_mat=np.meshgrid(rx_list,ry_list)

        val,vec,_=self.energy_single(kx,ky)

        psi1,psi2,psi3,psi4=self.psi(vec[:,state],rx_mat.flatten(),ry_mat.flatten(),kx,ky)
        psi1_2=np.abs(psi1.reshape(rx_mat.shape))**2
        psi2_2=np.abs(psi2.reshape(rx_mat.shape))**2
        psi3_2=np.abs(psi3.reshape(rx_mat.shape))**2
        psi4_2=np.abs(psi4.reshape(rx_mat.shape))**2
        psi1_arg=np.angle(psi1.reshape(rx_mat.shape))
        psi2_arg=np.angle(psi2.reshape(rx_mat.shape))
        psi3_arg=np.angle(psi3.reshape(rx_mat.shape))
        psi4_arg=np.angle(psi4.reshape(rx_mat.shape))

        if ax is None:
            k=0.03
            fig,ax=plt.subplots(2,4,figsize=(k*(rx_list[-1]-rx_list[0])*(4+0.2*3),k*(ry_list[-1]-ry_list[0])*(2+0.2)),gridspec_kw=dict(wspace=0.2,hspace=0.2))
        [self.plot_lattice(nshell=2,ax=ax) for ax in ax.flatten()]
        im_abs=[ax.pcolormesh(rx_list/self.aM,ry_list/self.aM,psi,cmap='Blues',shading='auto',vmin=0) for ax,psi in zip(ax[0],[psi1_2,psi2_2,psi3_2,psi4_2])]
        im_arg=[ax.pcolormesh(rx_list/self.aM,ry_list/self.aM,psi,cmap='hsv',shading='auto',vmin=-np.pi,vmax=np.pi) for ax,psi in zip(ax[1],[psi1_arg,psi2_arg,psi3_arg,psi4_arg])]


        [ax.set_xlabel('$x/|a_M|$') for ax in ax[1]]
        [ax.set_ylabel('$y/|a_M|$') for ax in ax[:,0]]
        [ax.set_xticklabels([]) for ax in ax[0]]
 
        axins=[ax.inset_axes([.5,1.05,.5,.05],transform=ax.transAxes) for ax in ax.flatten()]
        cb=[plt.colorbar(im,cax=axins,orientation='horizontal') for im,axins in zip(im_abs+im_arg,axins)]

        [axins.xaxis.set_label_position('top') for axins in axins]
        [axins.xaxis.set_ticks_position('top') for axins in axins]
        [axins.xaxis.set_tick_params(pad=0) for axins in axins]
        [ax.text(0,1.02,r'$|\psi_k^{{({})}}(r)|^2$'.format(layer),transform=ax.transAxes) for layer,ax in zip('1234',ax[0])]
        [ax.text(0,1.02,r'Arg($\psi_k^{{({})}}(r)$)'.format(layer),transform=ax.transAxes) for layer,ax in zip('1234',ax[1])]
        [ax.set_xlim(rx_list[0]/self.aM,rx_list[-1]/self.aM) for ax in ax.flatten()]
        [ax.set_ylim(ry_list[0]/self.aM,ry_list[-1]/self.aM) for ax in ax.flatten()]
        return psi1,psi2,psi3,psi4

        # return psib2,psit2,psib_arg,psit_arg,rx_mat,ry_mat

    def plot_bandstructure(self,state,ax=None,bz='hex',vH=None,FS=['dashed','dashed'],FS_shift=[0.02,0.02]):
        if not hasattr(self, 'energy_list'):
            energy_list=[]
            for kx,ky in (self.high_symm[bz]):
                val,vec,_=self.energy_single(kx, ky)
                energy_list.append(val)
            energy_list=np.array(energy_list)
            self.energy_list=energy_list
        else:
            energy_list=self.energy_list


        if bz=='hex':
            l,r,b,t=-params.high_symm['kappa'][0],params.high_symm['kappa'][0],-params.high_symm['kappa_p'][1],params.high_symm['kappa_p'][1]
        elif bz=='diamond':
            l,r,b,t=0,2*params.bM2[0],-params.bM2[1],params.bM2[1]
        if ax is not False:
            if ax is None:
                fig,ax=plt.subplots(figsize=(4,4*(t-b)/(r-l)))
            energy_list_bz=energy_list[:,state]
            if vH is not None:
                levels=np.r_[np.linspace(energy_list_bz.min()+FS_shift[0],energy_list_bz.max()-FS_shift[1],2),vH]
                ls=np.array([FS[0]]+[FS[-1]]+['dotted'])
                colors=np.array(['r','r','k'])
            else:
                levels=np.linspace(energy_list_bz.min()+FS_shift[0],energy_list_bz.max()-FS_shift[1],2)
                # levels=np.linspace(energy_list_bz.min()*1.1,energy_list_bz.max()*.9,2)
                ls=np.array([FS[0]]+[FS[-1]])
                colors=np.array(['r','r'])
            ls=ls[levels.argsort()]
            colors=colors[levels.argsort()]
            levels=np.sort(levels)

            im=ax.tripcolor(*(self.high_symm[bz].T/self.bM),energy_list_bz,cmap='Blues')
            ax.tricontour(*(self.high_symm[bz].T/self.bM),energy_list_bz,levels=levels,linestyles=ls,colors=colors)
            ax.set_xlabel('$k_x/|b_M|$')
            ax.set_ylabel('$k_y/|b_M|$')
            ax.set_xlim(l/self.bM,r/self.bM)
            ax.set_ylim(b/self.bM,t/self.bM)
            ax.text(0.5,1.02,'E (meV)',transform=ax.transAxes,ha='right')
            ax.text(0.,1.02,'$E_z$={}'.format(self.Ez),transform=ax.transAxes,ha='left')

            axins=ax.inset_axes([.5,1.05,.5,.05],transform=ax.transAxes)
            cb=plt.colorbar(im,cax=axins,orientation='horizontal')
            axins.xaxis.set_label_position('top')
            axins.xaxis.set_ticks_position('top')
            axins.xaxis.set_tick_params(pad=0)
        
        return energy_list

    def plot_LDOS(self,ax=None,bw_method=0.15,state=2):
        if not hasattr(self, 'energy_list'):
            self.energy_list=self.plot_bandstructure(state=0,bz='diamond',ax=False)
        energy_map=self.energy_list[:,:state].flatten()
        kde=KernelDensity(kernel='exponential',bandwidth=bw_method).fit(energy_map[:,np.newaxis])
        energy_range=np.linspace(energy_map.min(), energy_map.max(),1001)
        filling_range=-np.sum((energy_range[:,np.newaxis]-energy_map)<0,axis=1)/energy_map.shape[0]*state
        log_dos=kde.score_samples(energy_range[:,np.newaxis])
        dos=np.exp(log_dos)/(np.sqrt(3)/2*(self.aM/10)**2)*1e3
        if ax is not False:
            if ax is None:
                fig,ax=plt.subplots(1,2,figsize=(8,4))
            ax[0].plot(energy_range,dos)
            ax[1].plot(filling_range,dos)
            ax[0].set_xlabel('E (meV)')
            ax[1].set_xlabel(r'$\nu$')
            ax[0].set_ylabel('DOS (eV$^{-1}$nm$^{-2}$)')
        return energy_range,filling_range,dos,energy_map

def deg2rad(theta):
    return theta*np.pi/180

def _rotate(theta):
    return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])

def generate_shell(nshell,kind='hex'):
    '''kind: hex (onsite)/ tri (center of the center of triangle)
    '''            
    if kind=='hex':
        neighborlist=[[xindex,yindex] for yindex in range(-nshell,nshell+1) for xindex in range(max(-nshell,-nshell+yindex),min(nshell+yindex,nshell)+1)]
    elif kind=='tri':
        neighborlist=[[xindex,yindex] for xindex in range(-nshell,1+2*nshell+1) for yindex in range(xindex-nshell,1+nshell+1)]
    return np.array(neighborlist)



def _interpolate_path(path,n):
    '''
    path: directional path
    n: # of pts 
    '''
    k=np.arange(n)/n 
    
    return np.hstack([(1-k)*start[0]+k*end[0] for start,end in zip(path[:-1],path[1:])]+[path[-1][0]]),np.hstack([(1-k)*start[1]+k*end[1] for start,end in zip(path[:-1],path[1:])]+[path[-1][1]])
