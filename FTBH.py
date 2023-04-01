import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix

class Monolayer:
    def __init__(self,material,data_TB,data_lattice,data_SOC=None,n=40,offset=0):
        '''
        N_n: distance of farthest neighbhors
        '''
        self.material=material
        self.data_SOC=data_SOC
        self.data_TB=data_TB
        self.M,self.X=self._get_element()
        self.n=n
        self.offset=offset
        
        self.a=data_lattice.loc['a',material]    #A
        self.a1=self.a*np.array([1,0])
        self.a2=self.a*np.array([np.cos(deg2rad(120)),np.sin(deg2rad(120))])
        self.b=4*np.pi/(np.sqrt(3)*self.a)
        self.g=[np.array([np.cos(deg2rad(x+90)),np.sin(deg2rad(x+90))])*self.b for x in np.arange(6)*60]
        self.b1=self.g[5]
        self.b2=self.g[0]
        self.t=self._convert_dict(self.material)
        self._extend_t()
        self.delta=self._generate_delta()
        self.high_symm=self._generate_high_symm()
        
        
        self.M1=np.diag([-1,-1,-1,-1,-1,1,1,1,1,1,1])
        self.M2=np.diag([-1,1,1,-1,1,1,-1,1,1,-1,1])
        self.M3=np.diag([1,-1,1,1,-1,1,-1,1,1,1,-1])
        
        
        
    def _get_element(self):
        mat=re.match(r'([A-Z][a-z]*)([A-Z][a-z]*)\d', self.material)
        M,X=mat.group(1,2)
        return M,X

    def _generate_high_symm(self):
        K_=(2*self.b1-self.b2)/3
        Kp_=(self.b1+self.b2)/3
        M_=self.b1/2
        G_=0*self.b1

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
        k_hex=u_index@np.array([self.b1,self.b2])/np.sqrt(3)@_rotate(deg2rad(30))

        ux,uy=np.mgrid[0:self.n,0:self.n]
        u_index=np.c_[ux.flatten()/self.n,uy.flatten()/self.n]
        k_diamond=u_index@np.array([self.b1,self.b2])
        
        return {'kappa':K_,'M':M_,'Gamma':G_,'kappa_p':Kp_,
        'G_M_K_G':(G_M_K_G_kx,G_M_K_G_ky),'G_M_K_G_dist':G_M_K_G_dist,'G_M_K_G_name':G_M_K_G_name,
        'K_M_Kp_G':(K_M_Kp_G_kx,K_M_Kp_G_ky),'K_M_Kp_G_dist':K_M_Kp_G_dist,'K_M_Kp_G_name':K_M_Kp_G_name,
        'hex':k_hex,'diamond':k_diamond,
        }
    
    def _generate_delta(self):
        return {1:self.a1,2:(self.a1+self.a2),3:self.a2,4:-(2*self.a1+self.a2)/3,5:(self.a1+self.a2*2)/3,6:(self.a1-self.a2)/3,7:-2*(self.a1+2*self.a2)/3,8:2*(2*self.a1+self.a2)/3,9:2*(self.a2-self.a1)/3}
    
    

    def _convert_dict(self,material):
        t={}
        for key,val in zip(self.data_TB['tb'],self.data_TB[material]):
            if 'epsilon_' in key:
                idx=int(key.replace('epsilon_',''))
                t[(0,idx,idx)]=float(val)
            if 't' in key:
                num=re.match(r't\^\((\d+)\)_(\d+),(\d+)',key)
                kind,i,j=num.group(1,2,3)
                t[(int(kind),int(i),int(j))]=float(val)
        return t

    def _extend_t(self):
        indices=[(1,2,None),(4,5,3),(7,8,6),(10,11,9)]
        for idx in indices:
            a,b,g=idx
            self.t[(2,a,a)]=self.t[(1,a,a)]/4+self.t[(1,b,b)]/4*3
            self.t[(2,b,b)]=self.t[(1,a,a)]/4*3+self.t[(1,b,b)]/4
            if g is not None:
                self.t[(2,g,g)]=self.t[(1,g,g)]
                self.t[(2,g,b)]=self.t[(1,g,a)]*np.sqrt(3)/2-self.t[(1,g,b)]/2
                self.t[(3,g,b)]=-self.t[(1,g,a)]*np.sqrt(3)/2-self.t[(1,g,b)]/2
                self.t[(2,g,a)]=self.t[(1,g,a)]/2+self.t[(1,g,b)]/2*np.sqrt(3)
                self.t[(3,g,a)]=self.t[(1,g,a)]/2-self.t[(1,g,b)]/2*np.sqrt(3)
            self.t[(2,a,b)]=(self.t[(1,a,a)]-self.t[(1,b,b)])*np.sqrt(3)/4-self.t[(1,a,b)]
            self.t[(3,a,b)]=-(self.t[(1,a,a)]-self.t[(1,b,b)])*np.sqrt(3)/4-self.t[(1,a,b)]

        indices=[(1,2,4,5,3),(7,8,10,11,9)]
        for idx in indices:
            a,b,ap,bp,gp=idx
            self.t[(4,ap,a)]=self.t[(5,ap,a)]/4+self.t[(5,bp,b)]/4*3
            self.t[(4,bp,b)]=self.t[(5,ap,a)]/4*3+self.t[(5,bp,b)]/4
            self.t[(4,bp,a)]=-np.sqrt(3)/4*self.t[(5,ap,a)]+np.sqrt(3)/4*self.t[(5,bp,b)]
            self.t[(4,ap,b)]=self.t[(4,bp,a)]
            self.t[(4,gp,a)]=-np.sqrt(3)/2*self.t[(5,gp,b)]
            self.t[(4,gp,b)]=-self.t[(5,gp,b)]/2

        self.t[(4,9,6)]=self.t[(5,9,6)]
        self.t[(4,10,6)]=-np.sqrt(3)/2*self.t[(5,11,6)]
        self.t[(4,11,6)]=-self.t[(5,11,6)]/2

    def get_Hamiltonian_monolayer_spinless(self,kx,ky,XM2=True,eff=None):
        if eff is not None:
            warnings.warn('eff will be ignored in monolayer')
        k=np.array([kx,ky])
        diag=[[i,i,self.offset+self.t[(0,i,i)]+2*self.t[(1,i,i)]*np.cos(k@self.delta[1])+2*self.t[(2,i,i)]*(np.cos(k@self.delta[2])+np.cos(k@self.delta[3]))] for i in range(1,12)]
        MM_pos=[[i,j,2*self.t[(1,i,j)]*np.cos(k@self.delta[1])+self.t[(2,i,j)]*(np.exp(-1j*k@self.delta[2])+np.exp(-1j*k@self.delta[3]))+self.t[(3,i,j)]*(np.exp(1j*k@self.delta[2])+np.exp(1j*k@self.delta[3]))] for i,j in [(3,5),(6,8),(9,11)]]
        MM_neg=[[i,j,-2j*self.t[(1,i,j)]*np.sin(k@self.delta[1])+self.t[(2,i,j)]*(np.exp(-1j*k@self.delta[2])-np.exp(-1j*k@self.delta[3]))+self.t[(3,i,j)]*(-np.exp(1j*k@self.delta[2])+np.exp(1j*k@self.delta[3]))] for i,j in [(1,2),(3,4),(4,5),(6,7),(7,8),(9,10),(10,11)]]
        MX_pos=[[i,j,self.t[(4,i,j)]*(np.exp(1j*k@self.delta[4])-np.exp(1j*k@self.delta[6]))] for i,j in [(3,1),(5,1),(4,2),(10,6),(9,7),(11,7),(10,8)]]
        MX_neg=[[i,j,self.t[(4,i,j)]*(np.exp(1j*k@self.delta[4])+np.exp(1j*k@self.delta[6]))+self.t[(5,i,j)]*np.exp(1j*k@self.delta[5])] for i,j in [(4,1),(3,2),(5,2),(9,6),(11,6),(10,7),(9,8),(11,8)]]
        if XM2:
            XM2_list=[
            [9,6,self.t[(6,9,6)]*(np.exp(1j*k@self.delta[7])+np.exp(1j*k@self.delta[8])+np.exp(1j*k@self.delta[9]))],
            [11,6,self.t[(6,11,6)]*(np.exp(1j*k@self.delta[7])-np.exp(1j*k@self.delta[8])/2-np.exp(1j*k@self.delta[9])/2)],
            [10,6,np.sqrt(3)/2*self.t[(6,11,6)]*(-np.exp(1j*k@self.delta[8])+np.exp(1j*k@self.delta[9]))],
            [9,8,self.t[(6,9,8)]*(np.exp(1j*k@self.delta[7])-1/2*np.exp(1j*k@self.delta[8])-1/2*np.exp(1j*k@self.delta[9]))],
            [9,7,np.sqrt(3)/2*self.t[(6,9,8)]*(-np.exp(1j*k@self.delta[8])+np.exp(1j*k@self.delta[9]))],
            [10,7,3/4*self.t[(6,11,8)]*(np.exp(1j*k@self.delta[8])+np.exp(1j*k@self.delta[9]))],
            [11,7,np.sqrt(3)/4*self.t[(6,11,8)]*(np.exp(1j*k@self.delta[8])-np.exp(1j*k@self.delta[9]))],
            [10,8,np.sqrt(3)/4*self.t[(6,11,8)]*(np.exp(1j*k@self.delta[8])-np.exp(1j*k@self.delta[9]))],
            [11,8,self.t[(6,11,8)]*(np.exp(1j*k@self.delta[7])+np.exp(1j*k@self.delta[8])/4+np.exp(1j*k@self.delta[9])/4)],
            ]
        else:
            XM2_list=[]

        off_diag=MM_pos+MM_neg+MX_pos+MX_neg+XM2_list

        H=sparse(diag)+sparse(off_diag)+sparse(off_diag).T.conj()
        H=H.toarray()
        return H
    


    def get_Hamiltonian_monolayer_spinful(self,kx,ky,XM2=True,eff=None):
        '''include spin-orbit coupling
        '''
        if eff is not None:
            warnings.warn('eff will be ignored in monolayer')
        assert self.data_SOC is not None, 'spin-orbit coupling is required with spinful Hamiltonian'
        H_1L=self.get_Hamiltonian_monolayer_spinless(kx, ky,XM2)
        H_SOC=self.get_SOC()
        H=np.kron(H_1L,np.eye(2))+H_SOC
        return H
    
    def get_SOC(self):
        r_M=[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 10, 10, 11, 11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 15, 15, 15]
        c_M=[2, 11, 13, 15, 3, 10, 12, 14, 0, 11, 13, 15, 1, 10, 12, 14, 1, 3, 0, 2, 1, 3, 14, 0, 2, 15, 1, 3, 12, 0, 2, 13]
        v_M=[-1j/2, np.sqrt(3)/4, 1j/4, -1/4, 1j/2, -np.sqrt(3)/4, 1j/4, 1/4, 1j/2, -np.sqrt(3)*1j/4, -1/4, -1j/4, -1j/2, -np.sqrt(3)*1j/4, 1/4, -1j/4, -np.sqrt(3)/4, np.sqrt(3)*1j/4, np.sqrt(3)/4, np.sqrt(3)*1j/4, -1j/4, 1/4, 1j, -1j/4, -1/4, -1j, 1/4, 1j/4, -1j, -1/4, 1j/4, 1j]

        r_A=[4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21]
        c_A=[7, 9, 19, 21, 6, 8, 18, 20, 5, 8, 17, 20, 4, 9, 16, 21, 5, 6, 17, 18, 4, 7, 16, 19, 7, 9, 19, 21, 6, 8, 18, 20, 5, 8, 17, 20, 4, 9, 16, 21, 5, 6, 17, 18, 4, 7, 16, 19]
        v_A=[-1/8, 1j/8, -1/8, 1j/8, 1/8, 1j/8, 1/8, 1j/8, 1/8, -1j/4, 1/8, -1j/4, -1/8, 1j/4, -1/8, 1j/4, -1j/8, 1j/4, -1j/8, 1j/4, -1j/8, -1j/4, -1j/8, -1j/4, -1/8, 1j/8, -1/8, 1j/8, 1/8, 1j/8, 1/8, 1j/8, 1/8, -1j/4, 1/8, -1j/4, -1/8, 1j/4, -1/8, 1j/4, -1j/8, 1j/4, -1j/8, 1j/4, -1j/8, -1j/4, -1j/8, -1j/4]

        r_B=[4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21]
        c_B=[7, 9, 19, 21, 6, 8, 18, 20, 5, 8, 17, 20, 4, 9, 16, 21, 5, 6, 17, 18, 4, 7, 16, 19, 7, 9, 19, 21, 6, 8, 18, 20, 5, 8, 17, 20, 4, 9, 16, 21, 5, 6, 17, 18, 4, 7, 16, 19]
        v_B=[1/8, -1j/8, -1/8, 1j/8, -1/8, -1j/8, 1/8, 1j/8, -1/8, -1j/4, 1/8, 1j/4, 1/8, 1j/4, -1/8, -1j/4, 1j/8, 1j/4, -1j/8, -1j/4, 1j/8, -1j/4, -1j/8, 1j/4, -1/8, 1j/8, 1/8, -1j/8, 1/8, 1j/8, -1/8, -1j/8, 1/8, 1j/4, -1/8, -1j/4, -1/8, -1j/4, 1/8, 1j/4, -1j/8, -1j/4, 1j/8, 1j/4, -1j/8, 1j/4, 1j/8, -1j/4]

        SOC_sparse=self.data_SOC[self.M]*csc_matrix((v_M,(r_M,c_M)),(22,22))+self.data_SOC[self.X]*(csc_matrix((v_A,(r_A,c_A)),(22,22))+csc_matrix((v_B,(r_B,c_B)),(22,22)))
        return SOC_sparse.toarray()

    
    

    def plot_dispersion(self,func,ax=None,line='G_M_K_G',color='k',XM2=True,eff=None,label=None):
        assert line in {'G_M_K_G','K_M_Kp_G'}, 'line should be "K_M_Kp_G" or "G_M_K_G"'
        energy_list=[]
        for kx,ky in zip(*self.high_symm[line]):
            H=func(kx,ky,XM2=XM2,eff=eff)
            val,vec=get_energy(H)
            energy_list.append(val)

        energy_list=np.array(energy_list)
        if ax is not False:
            if ax is None:
                fig,ax=plt.subplots(figsize=(6,6/self.a1[0]*self.a2[1]))
            ax.plot(self.high_symm[line+'_dist'],energy_list,color=color)
            ax.plot([],[],color=color,label=label)

            xticks,xticklabels=[],[]
            for pos,name in self.high_symm[line+'_name'].items():
                ax.axvline(x=pos,ls='dashed',color='k',lw=1)
                xticks.append(pos)
                xticklabels.append(name)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)

            ax.set_ylabel(r'$E$ (eV)')
            ax.set_xlim(self.high_symm[line+'_dist'][[0,-1]])
        return energy_list


def get_energy(H):
    val,vec=np.linalg.eigh(H)
    return val[::-1],vec[:,::-1]

def sparse(entry):
    '''
    entry: [(i,j,val),..]
    '''
    entry=np.array(entry)
    i=entry[:,0].real.astype(int)-1
    j=entry[:,1].real.astype(int)-1
    val=entry[:,2]
    return csc_matrix((val,(i,j)))

    
def deg2rad(theta):
    return theta*np.pi/180        
def _interpolate_path(path,n):
    '''
    path: directional path
    n: # of pts 
    '''
    k=np.arange(n)/n 
    
    return np.hstack([(1-k)*start[0]+k*end[0] for start,end in zip(path[:-1],path[1:])]+[path[-1][0]]),np.hstack([(1-k)*start[1]+k*end[1] for start,end in zip(path[:-1],path[1:])]+[path[-1][1]])
def generate_shell(nshell,kind='hex'):
    '''kind: hex (onsite)/ tri (center of the center of triangle)
    '''            
    if kind=='hex':
        neighborlist=[[xindex,yindex] for yindex in range(-nshell,nshell+1) for xindex in range(max(-nshell,-nshell+yindex),min(nshell+yindex,nshell)+1)]
    elif kind=='tri':
        neighborlist=[[xindex,yindex] for xindex in range(-nshell,1+2*nshell+1) for yindex in range(xindex-nshell,1+nshell+1)]
    return np.array(neighborlist)
def _rotate(theta):
    return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])

    
class Multilayer(Monolayer):
    def __init__(self, material,data_TB, data_lattice,data_interlayer,data_SOC,theta_list,shift_M_list,z_list=None,n=40,N_n=5,threshold=5,material_list=None,offset_list=None):
        super().__init__(data_TB=data_TB, material=material, data_lattice=data_lattice,data_SOC=data_SOC,n=40)
        self.data_lattice=data_lattice
        self.theta_list=theta_list
        self.N_n=N_n
        self.threshold=threshold
        self.material_list=material_list
        self.offset_list=offset_list
        self.dXX=data_lattice.loc['dXX',material]    #A
        self.c=data_lattice.loc['c',material]    #A
        self.XA_list,self.XB_list,self.M_list=[],[],[]
        if z_list is None:
            z_list=[self.c/2]*len(self.theta_list)
        z_list=[0]+np.cumsum([0]+z_list)
        for theta,shift_M,z in zip([0]+self.theta_list,[0]+shift_M_list,[0]+z_list):
            XA_pos_list,XB_pos_list,M_pos_list=self._generate_site(theta=theta,shift_M=shift_M,z=z)
            self.XA_list.append(XA_pos_list)
            self.XB_list.append(XB_pos_list)
            self.M_list.append(M_pos_list)
        self.nu={bond:data_interlayer.loc['nu','{}-{}_{}'.format(self.X,self.X,bond)] for bond in ['pi','sigma']}
        self.R={bond:data_interlayer.loc['R','{}-{}_{}'.format(self.X,self.X,bond)] for bond in ['pi','sigma']}
        self.eta={bond:data_interlayer.loc['eta','{}-{}_{}'.format(self.X,self.X,bond)] for bond in ['pi','sigma']}

    def _generate_site(self,theta,shift_M=0,z=0):
        X_pos=np.array([0,0])
        M_pos=(X_pos+(X_pos+self.a2)+(X_pos+self.a1+self.a2))/3
        h1,h2=np.mgrid[-self.N_n:self.N_n+1,-self.N_n:self.N_n+1]
        shift_list=np.c_[h1.flatten(),h2.flatten()]@np.array([self.a1,self.a2])
        XA_pos_list=np.array([np.r_[_rotate(deg2rad(theta))@(X_pos+shift)+M_pos*shift_M,z+self.dXX/2] for shift in shift_list])
        XB_pos_list=np.array([np.r_[_rotate(deg2rad(theta))@(X_pos+shift)+M_pos*shift_M,z-self.dXX/2] for shift in shift_list])
        M_pos_list=np.array([np.r_[_rotate(deg2rad(theta))@(M_pos+shift)+M_pos*shift_M,z] for shift in shift_list])

        return XA_pos_list,XB_pos_list,M_pos_list



    def plot_lattice(self,ax=None,M=True,XA=True,XB=True,show_layer=None,scale=1):
        x_max,x_min=self.XA_list[0][:,0].max(),self.XA_list[0][:,0].min()
        y_max,y_min=self.XA_list[0][:,1].max(),self.XA_list[0][:,1].min()
        if ax is None:
            fig,ax=plt.subplots(figsize=((x_max-x_min)*0.2*scale,(y_max-y_min)*0.2*scale))
        if show_layer is None:
            show_layer=range(len(self.XA_list))
        s_list=[320,160,40,10]
        color_list={'XA':[(r,r,0) for r in np.linspace(.5,1,4)],'XB':[(0,g,0) for g in np.linspace(.5,1,4)],'M':[(0,0,b) for b in np.linspace(.5,1,4)]}
        if XA:
            for layer in show_layer:
                ax.scatter(self.XA_list[layer][:,0]/self.a,self.XA_list[layer][:,1]/self.a,s=s_list[layer],color=color_list['XA'][layer],zorder=self.XA_list[layer][0,2],label=r'$X_A^{}$'.format(layer))
        if XB:
            for layer in show_layer:
                ax.scatter(self.XB_list[layer][:,0]/self.a,self.XB_list[layer][:,1]/self.a,s=s_list[layer],color=color_list['XB'][layer],zorder=self.XB_list[layer][0,2],label=r'$X_B^{}$'.format(layer))
        if M:
            for layer in show_layer:
                ax.scatter(self.M_list[layer][:,0]/self.a,self.M_list[layer][:,1]/self.a,s=s_list[layer],color=color_list['M'][layer],zorder=self.M_list[layer][0,2],label=r'$M^{}$'.format(layer))
        ax.legend()
        ax.grid('on')
        ax.set_xlim(x_min/self.a,x_max/self.a)
        ax.set_ylim(y_min/self.a,y_max/self.a)
        ax.set_xlabel('x/a')
        ax.set_ylabel('y/a')

    def get_Hamiltonian_spinless(self, kx, ky,XM2=True,eff=1):
        '''
        Use C2z to obtain other layers
        '''
        
        H_1L=[self.get_Hamiltonian_monolayer_spinless(kx,ky,XM2=XM2)]
        
        H_1L_m=self.get_Hamiltonian_monolayer_spinless(-kx,-ky,XM2=XM2)
        M=self.M2@self.M3
        H_1L_C2z=M@H_1L_m@M.conj().T

        H_tunnel=[]
        for layer_idx,theta in enumerate(self.theta_list):
            if theta==0:
                H_1L.append(H_1L[0])
            else:
                H_1L.append(H_1L_C2z)
            H_tunnel.append(eff*self.get_tunneling(kx, ky,layers=[layer_idx,layer_idx+1]))
        if self.offset_list is not None:
            for layer_idx,offset in enumerate(self.offset_list):
                H_1L[layer_idx]=H_1L[layer_idx]+offset*np.eye(H_1L[layer_idx].shape[0])

        return _assembly_Hamiltonian(H_1L,H_tunnel)

    def get_Hamiltonian_spinful(self,kx, ky,XM2=True,eff=1):
        H_1L=[self.get_Hamiltonian_monolayer_spinful(kx,ky,XM2=XM2)]

        H_1L_m=self.get_Hamiltonian_monolayer_spinful(-kx,-ky,XM2=XM2)

        M=np.kron(self.M2@self.M3,1j*np.array([[1,0],[0,-1]]))
        H_1L_C2z=M@H_1L_m@M.conj().T
        H_tunnel=[]
        for layer_idx,theta in enumerate(self.theta_list):
            if theta==0:
                H_1L.append(H_1L[0])
            else:
                H_1L.append(H_1L_C2z)
            H_tunnel.append(eff*np.kron(self.get_tunneling(kx, ky,layers=[layer_idx,layer_idx+1]),np.eye(2)))
        if self.offset_list is not None:
            for layer_idx,offset in enumerate(self.offset_list):
                H_1L[layer_idx]=H_1L[layer_idx]+offset*np.eye(H_1L[layer_idx].shape[0])
        return _assembly_Hamiltonian(H_1L,H_tunnel)


    def get_Hamitonian_heterobilayer_spinless(self,kx,ky,XM2=True,eff=1):
        '''
        Use a fake lattice constant, ignore the lattice mismatch
        material_list should start from the second layer
        '''
        H_1L=[self.get_Hamiltonian_monolayer_spinless(kx,ky,XM2=XM2)]
        H_tunnel=[]
        for layer_idx,(theta,material) in enumerate(zip(self.theta_list,self.material_list)):
            mat=Monolayer(material, data_TB=self.data_TB, data_lattice=self.data_lattice)
            if theta==0:
                H_1L.append(mat.get_Hamiltonian_monolayer_spinless(kx, ky))                
            else:
                H_1L_m=mat.get_Hamiltonian_monolayer_spinless(-kx, -ky)
                M=self.M2@self.M3
                H_1L.append(M@H_1L_m@M.conj().T)
            H_tunnel.append(eff*self.get_tunneling(kx, ky,layers=[layer_idx,layer_idx+1]))
        if self.offset_list is not None:
            for layer_idx,offset in enumerate(self.offset_list):
                H_1L[layer_idx]=H_1L[layer_idx]+offset*np.eye(H_1L[layer_idx].shape[0])

        return _assembly_Hamiltonian(H_1L,H_tunnel)

    def get_Hamitonian_heterobilayer_spinful(self,kx,ky,XM2=True,eff=1):
        '''
        Use a fake lattice constant, ignore the lattice mismatch
        material_list should start from the second layer
        '''
        H_1L=[self.get_Hamiltonian_monolayer_spinful(kx,ky,XM2=XM2)]
        H_tunnel=[]
        for layer_idx,(theta,material) in enumerate(zip(self.theta_list,self.material_list)):
            mat=Monolayer(material, data_TB=self.data_TB, data_lattice=self.data_lattice,data_SOC=self.data_SOC)
            if theta==0:
                H_1L.append(mat.get_Hamiltonian_monolayer_spinful(kx, ky))                
            else:
                H_1L_m=mat.get_Hamiltonian_monolayer_spinful(-kx, -ky)
                M=np.kron(self.M2@self.M3,1j*np.array([[1,0],[0,-1]]))
                H_1L.append(M@H_1L_m@M.conj().T)
            H_tunnel.append(eff*np.kron(self.get_tunneling(kx, ky,layers=[layer_idx,layer_idx+1]),np.eye(2)))
        if self.offset_list is not None:
            for layer_idx,offset in enumerate(self.offset_list):
                H_1L[layer_idx]=H_1L[layer_idx]+offset*np.eye(H_1L[layer_idx].shape[0])

        return _assembly_Hamiltonian(H_1L,H_tunnel)

            
    def get_tunneling(self,kx,ky,layers):
        '''
        Only partial tunneling for <top|H|bottom>
        '''
        t_xyz=self.get_t_mat(kx,ky,layers)
        t_zxy=t_xyz[np.ix_([2,0,1],[2,0,1])]
        t11=np.diag([1,-1,-1])@t_zxy
        t12=np.diag([-1,1,1])@t_zxy
        H_tunnel=np.zeros((11,11),dtype=np.complex)
        H_tunnel[2:5,2:5]=t11/2
        H_tunnel[8:,2:5]=t12/2
        H_tunnel[2:5,8:]=t11/2
        H_tunnel[8:,8:]=t12/2
        return H_tunnel
    
        
    def get_t_mat(self,kx,ky,layers):
        bottom_X_list=self.XA_list[layers[0]]
        top_X_list=self.XB_list[layers[1]]
        min_idx=np.linalg.norm(bottom_X_list[:,:2],axis=1).argmin()
        pos_list,r_list=self.sort_neighbors(top_X_list-bottom_X_list[min_idx])   

        mask=(r_list<=self.threshold)
        r_list=r_list[mask]
        pos_list=pos_list[mask]
        t_r_list,expo_list=[],[]
        for pos,r in zip(pos_list,r_list):
            Vpp=self.get_Vpp(r)
            t_r_list.append((Vpp['sigma']-Vpp['pi'])/r**2*np.outer(pos,pos) + Vpp['pi']*np.eye(3))
            expo_list.append(np.exp(-1j*np.array([kx,ky])@np.array(pos[:2])))
        t_k=np.sum([t_r*expo for t_r,expo in zip(t_r_list,expo_list)],axis=0)
        return t_k

    def sort_neighbors(self,position_list):
        r_list=np.linalg.norm(position_list,axis=1)
        order=r_list.argsort()
        return position_list[order],r_list[order]

    def get_Vpp(self,r):
        Vpp={bond:self.nu[bond]*np.exp(-(r/self.R[bond])**self.eta[bond]) for bond in ['pi','sigma']}
        return Vpp

def _assembly_Hamiltonian(H_1L,H_tunnel):
    '''
    H_1L is put to diagonal term
    H_tunnel is attach to first lower off-diagonal
    '''
    H=0
    for layer_idx in range(len(H_1L)):
        diag_mat=np.zeros((len(H_1L),len(H_1L)))
        diag_mat[layer_idx,layer_idx]=1
        H=H+np.kron(diag_mat,H_1L[layer_idx])
    
    for layer_idx in range(len(H_tunnel)):
        diag_mat=np.zeros((len(H_1L),len(H_1L)))
        diag_mat[layer_idx+1,layer_idx]=1
        H=H+np.kron(diag_mat,H_tunnel[layer_idx])+np.kron(diag_mat,H_tunnel[layer_idx]).conj().T
    return H
