import gc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dask.array as da
class Params:
    def __init__(self,t,U,h,nu,aM0=1,n=21,hartree=True, fock=True):
        self.t=t
        self.U=U
        self.h=h
        self.nu=nu
        self.aM0=aM0
        self.n=n
        self.hartree=hartree
        self.fock=fock
        # original lattice
        self.aM=self.aM0*np.array([[np.cos(np.deg2rad(-120)),np.sin(np.deg2rad(-120))],[np.cos(np.deg2rad(-60)),np.sin(np.deg2rad(-60))]])
        self.bM0=4*np.pi/(np.sqrt(3)*self.aM0)
        self.g=[np.array([np.cos(np.deg2rad(x)),np.sin(np.deg2rad(x))])*self.bM0 for x in np.arange(6)*60+30]
        self.bM=np.array([self.g[5],self.g[1]])
        self.neighbor_list=self._generate_neighbor()
        self._generate_ansatz()
        self._generate_k_mesh()
        self.high_symm=self._generate_high_symm()
        self.delta,self.delta12,self.expqq,=self._generate_delta() # q1,q2,q3,q4; q1,q2; n,q1,q2 
        self.expkk =self._generate_expkk() # n,k1,k2
        self.sigma=_generate_pauli_matrix()
        self.print_info()
        

    # Checked
    def _generate_neighbor(self):
        '''
        neighbor_list: unit cell
        '''
        neighbor_list={}
        neighbor_list[0]=np.array([self.aM[0]*0])
        neighbor_list[1]=np.array([self.aM[0]*0,self.aM[0],self.aM[1]])
        # neighbor_list[1]=np.array([-1/3*(self.aM[0]+self.aM[1]),1/3*(2*self.aM[0]-self.aM[1]),1/3*(2*self.aM[1]-self.aM[0])])
        neighbor_list[2]=np.array([self.aM[0],self.aM[1],self.aM[1]-self.aM[0],-self.aM[0],-self.aM[1],-self.aM[1]+self.aM[0]])
        # neighbor_list[2]=np.array([self.aM[0],self.aM[1],self.aM[1]-self.aM[0],-self.aM[0],-self.aM[1],-self.aM[1]+self.aM[0]])
        return neighbor_list

    # Checked
    def _generate_k_mesh(self):
        R=np.array([[0,-1],[1,0]])
        self.bm_index=1/(self.ai_index).shape[0]*R.T@self.am_index@R
        self.am=self.am_index@self.aM # am_index=[[h1,h2],[k1,k2]], aM=[aM1;aM2]
        self.bm=self.bm_index@self.bM # same as above
        ux,uy=np.meshgrid(np.arange(self.n),np.arange(self.n),indexing='ij')
        kx=(2*ux.flatten(order='F')-self.n+1)/(self.n*2)
        ky=(2*uy.flatten(order='F')-self.n+1)/(self.n*2)
        self.k_index=np.column_stack((kx,ky))
        self.k=self.k_index@self.bm
        self.q=self.q_index@self.bM
        self.ai=self.ai_index@self.aM

    # Checked
    def _generate_high_symm(self):
        # Caveat: this is assuming the super cell is still triangular lattice
        K_=(self.bm[0]+2*self.bm[1])/3
        Kp_=(-self.bm[0]+self.bm[1])/3
        M_=self.bm[1]/2
        G_=0*self.bm[0]

        G_M_K_G_kx,G_M_K_G_ky,G_M_K_G_dist,G_M_K_G_name=generate_line([G_,M_,K_,G_], self.n, [r'$\Gamma$',r'M',r'$\kappa$',r'$\Gamma$']) 

        K_M_Kp_G_kx,K_M_Kp_G_ky,K_M_Kp_G_dist,K_M_Kp_G_name=generate_line([K_,M_,Kp_,G_], self.n, [r'$\kappa$',r'M',r'$\kappa^\prime$',r'$\Gamma$']) 

        K6_kx,K6_ky,K6_dist,K6_name=generate_line([rotation(60*x)@K_ for x in range(7)], self.n, [rf'$\kappa_{i+1}$' for i in range(7)]) 

        K3_kx,K3_ky,K3_dist,K3_name=generate_line([rotation(120*x)@K_ for x in range(4)], self.n, [rf'$\kappa_{i+1}$' for i in range(4)]) 
        

        return {'kappa':K_,'M':M_,'Gamma':G_,'kappa_p':Kp_,
        'G_M_K_G':np.column_stack((G_M_K_G_kx,G_M_K_G_ky)),'G_M_K_G_dist':G_M_K_G_dist,'G_M_K_G_name':G_M_K_G_name,
        'K_M_Kp_G':np.column_stack((K_M_Kp_G_kx,K_M_Kp_G_ky)),'K_M_Kp_G_dist':K_M_Kp_G_dist,'K_M_Kp_G_name':K_M_Kp_G_name,
        'K6':np.column_stack((K6_kx,K6_ky)),'K6_dist':K6_dist,'K6_name':K6_name,
        'K3':np.column_stack((K3_kx,K3_ky)),'K3_dist':K3_dist,'K3_name':K3_name,
        }

    def print_info(self):
        resolution=np.sqrt(2*np.pi/(np.sqrt(3)*1/4)/(self.q.shape[0]*self.k.shape[0]))*np.sqrt(3)/2*np.abs(self.t[1])
        print(f'Energy resolution: {resolution}',flush=True)

    def _generate_ansatz(self):
        if self.nu==[-6,-3]:
            # Charge modulation
            self.ai_index=np.array([[0,0],[1,0],[2,0]])
            self.spinA0=[[1,0,0,0],[1,0,0,0],[1,0,0,0]]
            self.spinB0=[[1,0,0,0],[1,0,0,0],[1,0,0,0]]
            self.am_index=np.array([[1,1],[2,-1]])
            self.q_index=np.array([[0,0],[2/3,1/3],[-2/3,-1/3]])
        if self.nu==[-4,-2]:
            # Charge modulation
            self.ai_index=np.array([[0,0]])
            self.spinA0=[[1,0,0,0]]
            self.spinB0=[[1,0,0,0]]
            self.am_index=np.eye(2) 
            self.q_index=np.array([[0,0]])
        if self.nu==[-2,-1]:
            # Unpolarized
            self.ai_index=np.array([[0,0]])
            self.spinA0=[[2,0,0,0]]
            self.spinB0=[[0,0,0,0]]
            self.am_index=np.eye(2) 
            self.q_index=np.array([[0,0]])
        
        if self.nu==[2,1]:
            self.ai_index=np.array([[0,0]])
            self.spinA0=[[1,1,0,0]]
            self.spinB0=[[1,-1,0,0]]
            self.am_index=np.eye(2) 
            self.q_index=np.array([[0,0]])
        if self.nu==[4,2]:
            self.ai_index=np.array([[0,0],[1,0],[2,0]])
            self.spinA0=[[1,1,0,0],[1,1,0,0],[1,1,0,0]]
            self.spinB0=[[1,-1,0,0],[1,-1,0,0],[1,-1,0,0]]
            self.am_index=np.array([[1,1],[2,-1]])
            self.q_index=np.array([[0,0],[2/3,1/3],[-2/3,-1/3]])

        if self.nu==[-12,-6]:
            # Intravalley s wave
            self.ai_index=np.array([[0,0],[1,0],[2,0]])
            # self.spinA0=[[1,1,0,0],[1,1,0,0],[1,1,0,0]]
            # self.spinB0=[[1,-1,0,0],[1,-1,0,0],[1,-1,0,0]]
            self.am_index=np.array([[1,1],[2,-1]])
            # self.q_index=np.array([[0,0],[1/3,2/3],[2/3,1/3]])
            self.q_index=np.array([[0,0],[2/3,1/3],[-2/3,-1/3]])
            self.O_max=np.array([[40,0],[0,40]])
            self.l=np.zeros((2,2))
        if self.nu==[12,6]:
            # Intravalley s wave
            self.ai_index=np.array([[0,0],[1,0],[2,0]])
            # self.spinA0=[[1,1,0,0],[1,1,0,0],[1,1,0,0]]
            # self.spinB0=[[1,-1,0,0],[1,-1,0,0],[1,-1,0,0]]
            self.am_index=np.array([[1,1],[2,-1]])
            # self.q_index=np.array([[0,0],[1/3,2/3],[2/3,1/3]])
            self.q_index=np.array([[0,0],[2/3,1/3],[-2/3,-1/3]])
            self.O_max=np.array([[160,0],[0,-160]])
            self.l=np.zeros((2,2))

        if self.nu==[14,7]:
            # Intervalley s wave
            self.ai_index=np.array([[0,0],[1,0],[2,0]])
            # self.spinA0=[[1,1,0,0],[1,1,0,0],[1,1,0,0]]
            # self.spinB0=[[1,-1,0,0],[1,-1,0,0],[1,-1,0,0]]
            self.am_index=np.array([[1,1],[2,-1]])
            # self.q_index=np.array([[0,0],[1/3,2/3],[2/3,1/3]])
            self.q_index=np.array([[0,0],[2/3,1/3],[-2/3,-1/3]])
            self.O_max=np.array([[0,20],[20,0]])
            # self.O_max=np.array([[0,0],[160,0]])
            self.l=np.zeros((2,2))

        if self.nu==[16,8]:
            # Intravalley p wave
            self.ai_index=np.array([[0,0],[1,0],[2,0]])
            # self.spinA0=[[1,1,0,0],[1,1,0,0],[1,1,0,0]]
            # self.spinB0=[[1,-1,0,0],[1,-1,0,0],[1,-1,0,0]]
            self.am_index=np.array([[1,1],[2,-1]])
            # self.q_index=np.array([[0,0],[1/3,2/3],[2/3,1/3]])
            self.q_index=np.array([[0,0],[2/3,1/3],[-2/3,-1/3]])
            
            self.O_max=np.array([[-160,0],[0,-160]])
            self.l=np.diag([-1,1])
            # self.O_max=np.array([[160,0],[0,-160]])
            # self.l=np.ones((2,2))

        if self.nu==[18,9]:
            # Intervalley p wave
            self.ai_index=np.array([[0,0],[1,0],[2,0]])
            # self.spinA0=[[1,1,0,0],[1,1,0,0],[1,1,0,0]]
            # self.spinB0=[[1,-1,0,0],[1,-1,0,0],[1,-1,0,0]]
            self.am_index=np.array([[1,1],[2,-1]])
            # self.q_index=np.array([[0,0],[1/3,2/3],[2/3,1/3]])
            self.q_index=np.array([[0,0],[2/3,1/3],[-2/3,-1/3]])
            # self.O_max=np.array([[0,0],[160,0]])
            self.O_max=np.array([[0,160],[160,0]])
            # self.l=np.ones((2,2))
            self.l=np.array([[0,1],[1,0]])

        # if self.nu==[-18,-9]:
        #     # Intervalley p wave
        #     self.ai_index=np.array([[0,0],[1,0],[2,0]])
        #     # self.spinA0=[[1,1,0,0],[1,1,0,0],[1,1,0,0]]
        #     # self.spinB0=[[1,-1,0,0],[1,-1,0,0],[1,-1,0,0]]
        #     self.am_index=np.array([[1,1],[2,-1]])
        #     # self.q_index=np.array([[0,0],[1/3,2/3],[2/3,1/3]])
        #     self.q_index=np.array([[0,0],[2/3,1/3],[-2/3,-1/3]])
        #     # self.O_max=np.array([[0,0],[160,0]])
        #     self.O_max=np.array([[0,160],[160,0]])
        #     # self.l=np.ones((2,2))
        #     self.l=np.array([[0,1],[-1,0]])

        if self.nu==[6,3]:
            # SDW 120 Neel
            self.ai_index=np.array([[0,0],[1,0],[2,0]])
            self.spinA0=[[1,-1,0,0],[1,-np.cos(np.deg2rad(120)),-np.sin(np.deg2rad(120)),0],[1,-np.cos(np.deg2rad(240)),-np.sin(np.deg2rad(240)),0]]
            # self.spinB0=[[1,-1,0,0],[1,-np.cos(np.deg2rad(120)),-np.sin(np.deg2rad(120)),0],[1,-np.cos(np.deg2rad(240)),-np.sin(np.deg2rad(240)),0]]
            self.spinB0=[[1,0,0,0],[1,0,0,0],[1,0,0,0]]
            self.am_index=np.array([[1,1],[2,-1]])
            self.q_index=np.array([[0,0],[1/3,2/3],[2/3,1/3]])


        if self.nu==[10,5]:
            # SDW 120 Neel
            self.ai_index=np.array([[0,0],[1,0],[2,0]])
            self.spinA0=[[1,1,0,0],[1,np.cos(np.deg2rad(120)),np.sin(np.deg2rad(120)),0],[1,np.cos(np.deg2rad(240)),np.sin(np.deg2rad(240)),0]]
            # self.spinB0=[[1,-1*.1,0,0],[1,-np.cos(np.deg2rad(120))*.1,-np.sin(np.deg2rad(120))*.1,0],[1,-np.cos(np.deg2rad(240))*.1,-np.sin(np.deg2rad(240))*.1,0]]
            self.spinB0=[[1,0,0,0],[1,0,0,0],[1,0,0,0]]
            self.am_index=np.array([[1,1],[2,-1]])
            self.q_index=np.array([[0,0],[1/3,2/3],[2/3,1/3]])


        if self.nu==[8,4]:
            # triplet pairing
            self.ai_index=np.array([[0,0],[1,0],[2,0]])
            # self.spinA0=[[1,1,0,0],[1,-1/2,0,0],[1,-1/2,0,0]]
            # self.spinB0=[[1,-1,0,0],[1,1/2,0,0],[1,1/2,0,0]]
            self.spinB0=[[1,1,0,0],[1,np.cos(np.deg2rad(120)),np.sin(np.deg2rad(120)),0],[1,np.cos(np.deg2rad(240)),np.sin(np.deg2rad(240)),0]]
            self.spinA0=[[1,0,0,0],[1,0,0,0],[1,0,0,0]]
            self.am_index=np.array([[1,1],[2,-1]])
            self.q_index=np.array([[0,0],[1/3,2/3],[2/3,1/3]])
        
    # Obsolete
    def _ansatz_to_momentum(self):
        Nq,Nk=self.q.shape[0],self.k.shape[0]
        cc=np.zeros((Nq,2,2,2,2),dtype=complex) # a,s1,s2,α1,α2
        for ai_idx,ai in enumerate(self.ai_index):
            ccA=1/2*np.sum([self.spinA0[ai_idx][i]*self.sigma[i] for i in range(4)],axis=0).conj()
            ccB=1/2*np.sum([self.spinB0[ai_idx][i]*self.sigma[i] for i in range(4)],axis=0).conj()
            cc[ai_idx,:,:,0,0]=ccA
            cc[ai_idx,:,:,1,1]=ccB
        ai_x,q1_x,q2_x=np.meshgrid(self.ai[:,0],self.q[:,0],self.q[:,0],indexing='ij')
        ai_y,q1_y,q2_y=np.meshgrid(self.ai[:,1],self.q[:,1],self.q[:,1],indexing='ij')
        exp=np.exp(1j*((q1_x-q2_x)*ai_x+(q1_y-q2_y)*ai_y)) # a,q1,q2
        ave=np.tensordot(exp, cc,axes=([0],[0])) # q1,q2,s1,s2,α1,α2
        ave=np.tile(ave[:,:,:,:,:,:,np.newaxis],[1,1,1,1,1,1,Nk])/Nq # q1,q2,s1,s2,α1,α2, k
        ave=np.transpose(ave,axes=(6,0,2,4,1,3,5)) # k,q1,s1,α1, q2,s2,α2
        return ave
    
    # Checked
    def _ansatz_from_order_parameter(self,O_max,l=np.zeros((2,2))):
        '''
        O is a 4 by 4 matrix defined by 
        <+K-↑ , +K+↓> <+K-↑ , -K+↓>
        <-K-↑ , +K+↓> <-K-↑ , -K+↓>
        for h>0
        or 
        <+K-↓ , +K+↑> <+K-↓ , -K+↑>
        <-K-↓ , +K+↑> <-K-↓ , -K+↑>
        for h<0

        Return: ave_(k',q1,s1,α1, q2,s2,α2)
        '''
        Nq,Nk=self.q.shape[0],self.k.shape[0]
        theta_k=np.angle(self.k[:,0]+1j*self.k[:,1])
        norm_k=np.abs(self.k[:,0]+1j*self.k[:,1])
        exp_theta_k=np.exp(1j*theta_k)
        ave=np.zeros((Nk,Nq,2,2,Nq,2,2),dtype=complex)
        k_F=np.abs(np.abs(self.h)/(np.sqrt(3)/2*self.t[1]*self.aM0))
        for k_idx in range(ave.shape[0]):
            O=O_max* np.exp(-(norm_k[k_idx]-k_F)**2/(2*(k_F/4)**2))*np.exp(1j*l*theta_k[k_idx])
            # O=O_max*np.exp(1j*l*theta_k[k_idx])
            OO=np.array(
                [[O[0,0]/2, O[0,0]*exp_theta_k[k_idx]/2, -O[0,1]*exp_theta_k[k_idx]/2, O[0,1]/2],
                [-O[0,0]*exp_theta_k[k_idx].conj()/2,  -O[0,0]/2,  O[0,1]/2,  -O[0,1]*exp_theta_k[k_idx].conj()/2],
                [O[1,0]*exp_theta_k[k_idx].conj()/2,  O[1,0]/2,  -O[1,1]/2,  O[1,1]*exp_theta_k[k_idx].conj()/2],
                [O[1,0]/2,  O[1,0]*exp_theta_k[k_idx]/2,  -O[1,1]*exp_theta_k[k_idx]/2,  O[1,1]/2]])
            block=self.sigma['+'] if self.h>=0 else self.sigma['-']
            OOO=np.kron(block,OO)+np.kron(block,OO).conj().T # (s,q,α),(s,q,α)
            OOO_2=np.reshape(OOO,(2,2,2,2,2,2)) # s1,q1,α1,s2,q2,α2
            OOO_3=np.transpose(OOO_2,(1,0,2,4,3,5)) # q1,s1,α1,q2,s2,α2
            ave[k_idx,1:,:,:,1:,:,:]=OOO_3
        return ave
            
    # Revised
    def energy_MF(self,ave):
        '''
        T,H_H,H_F: k, (q1,s1,α1),(q2,s2,α2)
        '''
        T=self._generate_T(self.k)
        H_H=self._generate_Hartree(ave) if self.hartree else 0
        H_F=self._generate_Fock(ave) if self.fock else 0
        H=T+H_H+H_F
        energyall,wfall=self._eig(H)
        return energyall,wfall
    # Unchecked
    def iteration(self,print_opt='o',thres=1e-5,iter_max=300,plot=False):
        ave=self._ansatz_to_momentum() if not hasattr(self, 'O_max') else self._ansatz_from_order_parameter(self.O_max,self.l)
        gap_list=[]
        spin_list=[]
        energy_list=[]
        gap_list.append(np.nan)
        energy_list.append(np.nan)
        spin_list.append(self.print_spin(ave,print_opt))
        for i in range(iter_max):
            energyall,wfall=self.energy_MF(ave)
            ave=self._average(energyall, wfall)
            print(f'Iteration {i}:\t',end=None)
            gap_list.append(self.print_gap(energyall))
            energy_list.append(self.total_energy(ave))
            spin_list.append(self.print_spin(ave,print_opt))
            if plot:
                self.plot_dispersion(ave)
            if i>0 and np.abs(energy_list[-1]-energy_list[-2])<thres:
                break

        return np.array(gap_list),np.array(energy_list),np.array(spin_list),ave
    # Checked  
    def _eig(self,H):
        '''
        H: k, (q1,s1,α1),(q2,s2,α2)
        '''
        herr=np.abs(H-H.transpose((0,2,1)).conj()).max()
        assert herr<1e-12, f'hermittian error:{herr}'
        H=1/2*(H+H.transpose((0,2,1)).conj())
        Nk,Nq=H.shape[0],self.q.shape[0]
        energyall=np.zeros((Nk,Nq*4),dtype=float)   # k, l
        wfall=np.zeros((Nk,Nq*4,Nq*4),dtype=complex)    #k,l,l
        for k_idx in range(Nk):
            H0=H[k_idx,:,:]
            val,vec=np.linalg.eigh(H0)
            order=np.argsort(val)
            vec=vec[:,order]
            val=val[order]
            energyall[k_idx]=val
            wfall[k_idx]=vec.T
        return energyall,wfall
    # Checked 
    def _generate_T(self,k_list):
        '''
        T: k, (q1,s1,α1),(q2,s2,α2)
        '''
        Nq,Nk=self.q.shape[0],k_list.shape[0]
        T=np.zeros((Nk,2*2*Nq,2*2*Nq),dtype=complex)
        for k_idx,k in enumerate(k_list):
            T0=np.zeros((2*2*Nq,2*2*Nq),dtype=complex)
            for q_idx,q in enumerate(self.q):
                gamma1=np.sum(np.exp(1j*self.neighbor_list[1]@(q+k)))
                gamma2=np.sum(np.exp(1j*self.neighbor_list[2]@(q+k)))
                H_up=np.array([[self.t[0]+self.t[2]*gamma2+self.h, self.t[1]*gamma1 ],
                               [self.t[1]*gamma1.conj(), self.t[0]+self.t[2]*gamma2+self.h]])
                H_down=np.array([[self.t[0]+self.t[2]*gamma2-self.h, self.t[1]*gamma1 ],
                               [self.t[1]*gamma1.conj(), self.t[0]+self.t[2]*gamma2-self.h]])
                T0[4*q_idx:4*q_idx+2,4*q_idx:4*q_idx+2]=H_up
                T0[4*q_idx+2:4*q_idx+4,4*q_idx+2:4*q_idx+4]=H_down
            T[k_idx,:,:]=T0
        return T
    # Checked 
    def _generate_delta(self,thres=1e-10):
        q1_index_1,q2_index_1,q3_index_1,q4_index_1=np.meshgrid(self.q_index[:,0],self.q_index[:,0],self.q_index[:,0],self.q_index[:,0],indexing='ij')
        q1_index_2,q2_index_2,q3_index_2,q4_index_2=np.meshgrid(self.q_index[:,1],self.q_index[:,1],self.q_index[:,1],self.q_index[:,1],indexing='ij')
        delta_1=q1_index_1-q2_index_1+q3_index_1-q4_index_1
        delta_2=q1_index_2-q2_index_2+q3_index_2-q4_index_2
        delta=((np.abs(delta_1%1)<thres) & (np.abs(delta_2%1)<thres)).astype('int8') # q1,q2,q3,q4

        delta12=np.eye(self.q_index.shape[0]) # q1,q2

        n_x,q1_x,q2_x=np.meshgrid(self.neighbor_list[1][:,0],self.q[:,0],self.q[:,0],indexing='ij')
        n_y,q1_y,q2_y=np.meshgrid(self.neighbor_list[1][:,1],self.q[:,1],self.q[:,1],indexing='ij')
        expqq=np.exp(-1j*((q1_x-q2_x)*n_x+(q1_y-q2_y)*n_y)) # n,q1,q2
        
        return delta,delta12,expqq
    def _generate_expkk(self,k1_list=None,k2_list=None,chunks=None):
        '''
        exp(-i * (k1-k2) * n)

        '''
        if k1_list is None:
            k1_list=self.k
        if k2_list is None:
            k2_list=self.k
        
        if k1_list.shape[0]*k2_list.shape[0]>69**4:
            print(f'Using Dask because the array size is going to be {k1_list.shape[0]*k2_list.shape[0]}',flush=True)
            if chunks is None:
                chunks=69**2
            k1_list=da.from_array(k1_list,chunks=(chunks,2))
            k2_list=da.from_array(k2_list,chunks=(chunks,2))
            n_x, k1_x, k2_x = da.meshgrid(self.neighbor_list[1][:,0],k1_list[:,0],k2_list[:,0], indexing='ij')
            n_y, k1_y, k2_y = da.meshgrid(self.neighbor_list[1][:,1],k1_list[:,1],k2_list[:,1], indexing='ij')
            return da.exp(-1j * ((k1_x - k2_x) * n_x + (k1_y - k2_y) * n_y))
        else:
            n_x,k1_x,k2_x=np.meshgrid(self.neighbor_list[1][:,0],k1_list[:,0],k2_list[:,0],indexing='ij')
            n_y,k1_y,k2_y=np.meshgrid(self.neighbor_list[1][:,1],k1_list[:,1],k2_list[:,1],indexing='ij')
            return np.exp(-1j*((k1_x-k2_x)*n_x+(k1_y-k2_y)*n_y)) # n,k1,k2

    # Revised 
    def _generate_Hartree(self,ave,k_list=None):
        '''
        ave k',q3,s3,α3, q4,s4,α4
        k_p_* for summation in ave
        k_* for summation outside ave, i.e., used to construct basis
        '''
        if k_list is None:
            k_list=self.k
        k_p_list=self.k

        Nq,Nk_p,Nk=self.q.shape[0],k_p_list.shape[0],k_list.shape[0]
        # N=Nq*Nk_p
            
        ave_k=np.sum(ave,axis=0)   #q3,s3,α3,q4,s4,α4
        delta_ave=np.tensordot(self.delta, ave_k,axes=([2,3],[0,3]))  # q1,q2,s3,α3,s4,α4
        delta=np.tensordot(np.tensordot(np.tensordot(self.sigma[1],self.sigma[0],axes=0),self.sigma[1],axes=0),self.sigma[0],axes=0) # s3,s1,α3,α1,s4,s2,α4,α2
        delta_ave_delta=np.tensordot(delta_ave,delta,axes=([2,3,4,5],[0,2,4,6])) # q1,q2,s1,α1,s2,α2
        
        delta_ave_delta=delta_ave_delta*self.sigma[0][np.newaxis,np.newaxis,:,np.newaxis,:,np.newaxis]*self.sigma[0][np.newaxis,np.newaxis,np.newaxis,:,np.newaxis,:]
        delta_ave_delta=np.transpose(delta_ave_delta,axes=(0,2,3,1,4,5)) # q1,s1,α1,q2,s2,α2
        H_H_0=delta_ave_delta.reshape((Nq*4,Nq*4))*self.U[0]/(Nq*Nk_p)

        
        delta=np.tensordot(np.tensordot(self.sigma[0],self.sigma[1],axes=0),self.sigma[1],axes=0) # s3,s4,α3,α1,α4,α2
        delta_2=3*self.delta*self.delta12[np.newaxis,np.newaxis,:,:] # q1,q2,q3,q4
        delta_ave=np.tensordot(delta_2, ave_k,axes=([2,3],[0,3]))  # q1,q2,s3,α3,s4,α4
        delta_ave_delta_1=np.tensordot(delta_ave,delta,axes=([2,3,4,5],[0,2,1,4])) # q1,q2,α1,α2
        delta_ave_delta_1=np.tensordot(delta_ave_delta_1*self.sigma[0][np.newaxis,np.newaxis,:,:],self.sigma[0],axes=0) # q1,q2,α1,α2,s1,s2
        delta_ave_delta_1=np.transpose(delta_ave_delta_1,axes=(0,4,2,1,5,3)) # q1,s1,α1,q2,s2,α2
        H_H_1=delta_ave_delta_1.reshape((Nq*4,Nq*4))*self.U[1]/(Nq*Nk_p)
        
        H_H=H_H_0+H_H_1
        return np.tile(H_H[np.newaxis,:,:],[Nk,1,1])

    # Revised 
    def _generate_Fock(self,ave,k_list=None):
        '''
        ave k',q3,s3,α3, q4,s4,α4
        k_p_* for summation in ave
        k_* for summation outside ave, i.e., used to construct basis
        '''
        if k_list is None:
            k_list=self.k
            if self.U[1]!=0:
                expkk=self.expkk
        else:
            if self.U[1]!=0:
                expkk=self._generate_expkk(k_list,self.k)
        k_p_list=self.k
        Nq,Nk_p,Nk=self.q.shape[0],k_p_list.shape[0],k_list.shape[0]
        # N=Nq*Nk_p

        ave_k=np.sum(ave,axis=0)   #q3,s3,α3,q4,s4,α4
        delta_ave=np.tensordot(self.delta, ave_k,axes=([2,3],[0,3]))  # q1,q2,s3,α3,s4,α4
        delta=np.tensordot(np.tensordot(np.tensordot(self.sigma[0],self.sigma[0],axes=0),self.sigma[0],axes=0),self.sigma[0],axes=0) # s3,s2,α3,α1,s4,s1,α4,α2
        delta_ave_delta=np.tensordot(delta_ave,delta,axes=([2,3,4,5],[0,2,4,6])) # q1,q2,s2,α1,s1,α2
        delta_ave_delta=delta_ave_delta*self.sigma[1][np.newaxis,np.newaxis,:,np.newaxis,:,np.newaxis]*self.sigma[0][np.newaxis,np.newaxis,np.newaxis,:,np.newaxis,:]
        delta_ave_delta=np.transpose(delta_ave_delta,axes=(0,4,3,1,2,5)) # q1,s1,α1,q2,s2,α2
        H_F_0=-delta_ave_delta.reshape((Nq*4,Nq*4))*self.U[0]/(Nq*Nk_p)

        if self.U[1]==0:
            return np.tile(H_F_0[np.newaxis,:,:],[Nk,1,1])
        delta_exp14=self.expqq[:,:,np.newaxis,np.newaxis,:]*self.delta[np.newaxis,:,:,:,:] # n, q1,q2,q3,q4
        delta_exp23=self.expqq[:,np.newaxis,:,:,np.newaxis].conj()*self.delta[np.newaxis,:,:,:,:] # n, q1,q2,q3,q4
        # if k_list is None:
        #     expkk=self.expkk
        # else:
        #     expkk=self._generate_expkk(k_list,self.k)
        exp_ave=np.tensordot(expkk,ave,axes=([2],[0])) # n,k,q3,s3,α3, q4,s4,α4
        exp_conj_ave=np.tensordot(expkk.conj(),ave,axes=([2],[0])) # n,k,q3,s3,α3, q4,s4,α4

        delta_alpha=self.sigma[0][:,np.newaxis,np.newaxis,:]*self.sigma[0][np.newaxis,:,:,np.newaxis]*self.sigma[1][:,:,np.newaxis,np.newaxis] # α1,α2,α3,α4
        delta_A=np.array([1,0]) # α4
        delta_B=np.array([0,1]) # α4
        delta_alpha_A=delta_A[np.newaxis,np.newaxis,np.newaxis,:]*delta_alpha
        delta_alpha_B=delta_B[np.newaxis,np.newaxis,np.newaxis,:]*delta_alpha
        exp_ave_delta=np.tensordot(exp_ave,delta_alpha_B,axes=([3,6],[2,3])) # n,k,q3,s3,q4,s4,α1,α2
        exp_conj_ave_delta=np.tensordot(exp_conj_ave,delta_alpha_A,axes=([3,6],[2,3])) # n,k,q3,s3,q4,s4,α1,α2
        delta_exp_ave_delta=np.tensordot(delta_exp14,exp_ave_delta,axes=([0,3,4],[0,2,4])) # q1,q2,k,s3,s4,α1,α2
        delta_exp_conj_ave_delta=np.tensordot(delta_exp23,exp_conj_ave_delta,axes=([0,3,4],[0,2,4])) # q1,q2,k,s3,s4,α1,α2
        delta_exp_tot=delta_exp_ave_delta+delta_exp_conj_ave_delta # q1,q2,k,s3,s4,α1,α2
        delta_s=delta_alpha # s1,s2,s3,s4
        delta_exp=np.tensordot(delta_exp_tot,delta_s,axes=([3,4],[2,3])) # q1,q2,k, α1,α2, s1,s2
        delta_exp_T=np.transpose(delta_exp,axes=(2,0,5,3,1,6,4))
        H_F_1=-delta_exp_T.reshape((Nk,Nq*4,Nq*4))*self.U[1]/(Nq*Nk_p)

        if isinstance(exp_ave,da.Array):
            H_F_1=H_F_1.compute()

        H_F=H_F_0[np.newaxis,:,:]+H_F_1

        return H_F
        
    
    # Revised 
    def _average(self,energyall,wfall):
        Nq,Nk=self.q.shape[0],self.k.shape[0]
        N=Nq*Nk
        energyall_sort=np.sort(energyall.flatten())
        mu=energyall_sort[Nq*Nk*self.nu[0]//self.nu[1]-1] # !!Need check
        occ=(energyall<=mu)  # k2,l
        c=wfall.reshape((Nk,4*Nq,Nq,2,2))   # c_{k,l,q2,s2,α2}
        c_conj=c.conj()  # c_{k,l,q1,s1,α1}
        c_conj_c=c_conj[:,:,:,:,:,np.newaxis,np.newaxis,np.newaxis]*c[:,:,np.newaxis,np.newaxis,np.newaxis,:,:,:] # k,l,q1,s1,α1,q2,s2,α2
        c_conj_c_occ=c_conj_c*occ[:,:,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis,np.newaxis] # k,l,q1,s1,α1,q2,s2,α2
        ave=np.sum(c_conj_c_occ,axis=1) # k,q1,s1,α1,q2,s2,α2
        return ave



    # Revised 
    def total_energy(self,ave):
        '''
        total average energy per unit cell
        '''
        Nq,Nk=self.q.shape[0],self.k.shape[0]
        N=Nq*Nk
        # Kinetic
        T=self._generate_T(self.k) # This is redundant, change it to self.T
        T=T.reshape((Nk,Nq,2,2,Nq,2,2)) # k,q1,s1,α1,q2,s2,α2
        H_0=np.tensordot(ave, T,axes=([0,1,2,3,4,5,6],[0,1,2,3,4,5,6]))

        # Hatree 0
        delta=np.tensordot(self.sigma[0],self.sigma[0],axes=0)*self.sigma[0][:,np.newaxis,:,np.newaxis] #  δ_{α1,α2} ⊗  δ_{α3,α4}  * δ_{α1,α3} = δ_{α1,α2,α3,α4}
        ave_uu=ave[:,:,0,:,:,0,:].sum(axis=0) # q1,α1,q2,α2
        ave_dd=ave[:,:,1,:,:,1,:].sum(axis=0) # q3,α3,q4,α4
        ave_dd_delta=np.tensordot(ave_dd,delta,axes=([1,3],[2,3])) # q3,q4,α1,α2
        ave_uu_ave_dd_delta=np.tensordot(ave_uu,ave_dd_delta,axes=([1,3],[2,3])) # q1,q2,q3,q4
        delta_ave_uu_ave_dd_delta=np.tensordot(self.delta, ave_uu_ave_dd_delta,axes=([0,1,2,3],[0,1,2,3]))

        H_H_0=delta_ave_uu_ave_dd_delta*self.U[0]/N

        # Fock 0
        ave_du=ave[:,:,1,:,:,0,:].sum(axis=0) # q1,α1,q2,α2
        ave_ud=ave[:,:,0,:,:,1,:].sum(axis=0) # q3,α3,q4,α4
        ave_ud_delta=np.tensordot(ave_ud,delta,axes=([1,3],[2,3])) # q3,q4,α1,α2
        ave_du_ave_ud_delta=np.tensordot(ave_du,ave_ud_delta,axes=([1,3],[2,3])) # q1,q2,q3,q4
        delta_ave_du_ave_du_delta=np.tensordot(self.delta, ave_du_ave_ud_delta,axes=([0,1,2,3],[0,1,2,3]))
        H_F_0=-delta_ave_du_ave_du_delta*self.U[0]/N

        # Hartree 1 
        ave_AA=(ave[:,:,0,0,:,0,0]+ave[:,:,1,0,:,1,0]).sum(axis=(0)) # q1,q2
        ave_BB=(ave[:,:,0,1,:,0,1]+ave[:,:,1,1,:,1,1]).sum(axis=(0)) # q3,q4
        delta_2=3*self.delta*self.delta12[np.newaxis,np.newaxis,:,:] # q1,q2,q3,q4
        delta_ave_AA=np.tensordot(delta_2, ave_AA,axes=([0,1],[0,1])) # q3,q4
        delta_ave_AA_ave_BB=np.tensordot(delta_ave_AA,ave_BB,axes=([0,1],[0,1])) 
        H_H_1=delta_ave_AA_ave_BB*self.U[1]/N

        # Fock 1
        ave_AB=ave[:,:,:,0,:,:,1] # k,q1,s,q2,s'
        ave_BA=ave[:,:,:,1,:,:,0] # k',q3,s',q4,s
        exp_conj_ave_AB=np.tensordot(self.expkk.conj(),ave_AB,axes=([1],[0])) # n,k',q1,s,q2,s' # this may be redundent
        exp_conj_ave_ave=np.tensordot(exp_conj_ave_AB,ave_BA,axes=([1,3,5],[0,4,2])) # n,q1,q2,q3,q4
        delta_exp23=self.expqq[:,np.newaxis,:,:,np.newaxis].conj()*self.delta[np.newaxis,:,:,:,:] # n, q1,q2,q3,q4
        delta_exp23_exp_conj_ave_ave=np.tensordot(delta_exp23, exp_conj_ave_ave,axes=([0,1,2,3,4],[0,1,2,3,4]))
        H_F_1=-delta_exp23_exp_conj_ave_ave*self.U[1]/N

        # H_tot=H_F_0+H_F_1
        H_tot=H_0+H_H_0+H_H_1+H_F_0+H_F_1
        if isinstance(H_tot,da.Array):
            H_tot=H_tot.compute()
        assert np.abs(H_tot.imag)<1e-10, f"total energy is not real {H_tot}"
        H_tot=H_tot.real/N
        print(f'E(meV):{H_tot}')
        return H_tot
    # Unchecked 
    def print_gap(self,energyall):
        Nq,Nk=self.q.shape[0],self.k.shape[0]
        energyall_sort=np.sort(energyall.flatten())
        mu_v=energyall_sort[Nq*Nk*self.nu[0]//self.nu[1]-1]
        mu_c=energyall_sort[Nq*Nk*self.nu[0]//self.nu[1]]
        gap=mu_c-mu_v
        print(f'Gap(meV)={gap}')
        return gap

    # Checked     
    def _generate_spin_mat(self,ave,n=None):
        '''
        convert from ave (k1,q1,s1,α1,q2,s2,α2) to ave_exp (α1,α2,a1,a2) for each sigma matrix
        '''
        Nq,Nk=self.q.shape[0],self.k.shape[0]
        N=Nq*Nk
        if n is None:
            n=np.array([0,0])
        n_x,n_y=n
        k_x,q1_x,a1_x,q2_x,a2_x=np.meshgrid(self.k[:,0],self.q[:,0],self.ai[:,0],self.q[:,0],self.ai[:,0],indexing='ij')
        k_y,q1_y,a1_y,q2_y,a2_y=np.meshgrid(self.k[:,1],self.q[:,1],self.ai[:,1],self.q[:,1],self.ai[:,1],indexing='ij')
        exp=np.exp(-1j*(a1_x*(q1_x+k_x)+a1_y*(q1_y+k_y))+1j*((a2_x+n_x)*(q2_x+k_x)+(a2_y+n_y)*(q2_y+k_y))) # k1,q1,a1,q2,a2
        spin_mat=np.zeros((4,2,2,Nq,Nq),dtype=complex)
        for sigma_idx in range(4):
            ave_sigma=np.tensordot(ave,self.sigma[sigma_idx],axes=([2,5],[0,1])) # k,q1,α1,q2,α2
            spin_mat[sigma_idx]=np.tensordot(ave_sigma,exp,axes=([0,1,3],[0,1,3]))/N # α1,α2,a1,a2
        return spin_mat

    # Revised 
    def print_spin(self,ave,output='o'):
        Nq,Nk=self.q.shape[0],self.k.shape[0]
        spin_mat=self._generate_spin_mat(ave)

        spin_list=np.zeros((4,2*Nq))

        for spin_idx,spin in enumerate(spin_mat):
            spin_texture=np.transpose(spin,axes=(0,2,1,3)).reshape((2*Nq,2*Nq))
            if 'o' in output:
                # onsite
                spin_diag=np.diag(spin_texture)
                assert np.abs(spin_diag.imag).max()<1e-10, f"spin is not real for {spin_idx} with {spin_diag}"
                spin_str=' '.join([f'{spin:.4f}' for spin in spin_diag.real])
                print(f'S{spin_idx}: {spin_str}',flush=True)
                spin_list[spin_idx]=spin_diag.real
            if 'b' in output:
                # bond
                i,j=np.triu_indices(n=spin_texture.shape[0], k=1)
                spin_triu=spin_texture[i,j].flatten()
                spin_str=' '.join([f'{spin:.3f}' for spin in spin_triu])

                print(f'S{spin_idx}_b: {spin_str}')
            if 'm' in output and spin_idx==1:
                print((pd.DataFrame(np.round(np.transpose(spin,axes=(0,2,1,3)).reshape((2*Nq,2*Nq)),4),index=[f'{alpha}{idx}' for alpha in 'AB' for idx in range(1,1+Nq)],columns=[f'{alpha}{idx}' for alpha in 'AB' for idx in range(1,1+Nq)])).to_string(),flush=True)
        return spin_list.T

    def energy(self,ave,k_list):
        '''
        Calculate the energy for a specific set of points
        ave: 
        k_list:
        '''
        T=self._generate_T(k_list)
        H_H=self._generate_Hartree(ave,k_list) if self.hartree else 0
        H_F=self._generate_Fock(ave,k_list) if self.fock else 0
        H=T+H_H+H_F
        energy_list,wfall=self._eig(H)
        return energy_list,wfall

    # Revised 
    def plot_dispersion(self,ave=None,ax=None,line='G_M_K_G',**kwargs):
        '''
        '''
        assert line in {'G_M_K_G','K_M_Kp_G','K6','K3'}, 'line should be "K_M_Kp_G" or "G_M_K_G" or "K6" or "K3"' 
        if ave is None:
            ave = np.zeros((self.k.shape[0],self.q.shape[0],2,2,self.q.shape[0],2,2))
        energy_list,wfall=self.energy(ave,self.high_symm[line])

        if ax is not False:
            if ax is None:
                # fig,ax=plt.subplots(figsize=(4,4/self.am[0,0]*self.am[1,1]))
                fig,ax=plt.subplots()
            ax.plot(self.high_symm[line+'_dist'],energy_list[:],**kwargs)

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
    
def _interpolate_path(path,n):
    '''
    path: directional path
    n: # of pts 
    '''
    k=np.arange(n)/n 
    
    return np.hstack([(1-k)*start[0]+k*end[0] for start,end in zip(path[:-1],path[1:])]+[path[-1][0]]),np.hstack([(1-k)*start[1]+k*end[1] for start,end in zip(path[:-1],path[1:])]+[path[-1][1]])


def _generate_o_matrix():
    o=np.zeros((4,4,4,4))
    for i in range(4):
        for j in range(4):
            o[i,j,i,j]=1
    return o

def _generate_pauli_matrix():
    sigma={}
    sigma[0]=np.eye(2)
    sigma[1]=np.array([[0,1],[1,0]])
    sigma[2]=np.array([[0,-1j],[1j,0]])
    sigma[3]=np.array([[1,0],[0,-1]])
    sigma['+']=np.array([[0,1],[0,0]])
    sigma['-']=np.array([[0,0],[1,0]])
    return sigma
sublattice=lambda x: 'A' if x==0 else 'B'
valley=lambda x: '$\Gamma$' if x==0 else '+K' if x==1 else '-K'

def gap_vs_U(x,params,epsilon=None,k=1):
    rho=4/(3*np.pi)/params.t[1]**2*np.abs(params.h)
    if epsilon is None:
        epsilon=params.h
    return 2*epsilon*np.exp(-k/(x*rho))
def gap_vs_U_indep(U,t1,h,epsilon,k=1):
    rho=4/(3*np.pi)/t1**2*np.abs(h)
    return 2*epsilon*np.exp(-k/(U*rho))
rotation= lambda x: np.array([[np.cos(np.deg2rad(x)),-np.sin(np.deg2rad(x))],[np.sin(np.deg2rad(x)),np.cos(np.deg2rad(x))]])

def generate_line(pts,n,label):
    # G_M_K_G=[G_,M_,K_,G_]
    pts_kx,pts_ky=_interpolate_path(pts,n)
    pts_dist=np.r_[0,np.cumsum(np.sqrt(np.diff(pts_kx)**2+np.diff(pts_ky)**2))]
    pts_name={pts_dist[idx*n]:name for idx,name in enumerate(label)}
    return pts_kx,pts_ky,pts_dist,pts_name
