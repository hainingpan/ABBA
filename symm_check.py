import numpy as np
# M1 z->-z
def check_M1_spinless(model,kx,ky):
    Ham=model.get_Hamiltonian_monolayer_spinless(kx, ky)
    M1=model.M1
    Ham2=M1@Ham@M1
    return (Ham-Ham2).__abs__().max()
def check_M1_spinful(model,kx,ky):
    Ham=model.get_Hamiltonian_monolayer_spinful(kx, ky)
    M1=np.kron(model.M1,np.array([[-1j,0],[0,1j]]))
    Ham2=M1@Ham@M1.conj().T
    return (Ham-Ham2).__abs__().max()

# M2 x->-x
def check_M2_spinless(model,kx,ky):
    Ham_p=model.get_Hamiltonian_monolayer_spinless(kx, ky)
    Ham_m=model.get_Hamiltonian_monolayer_spinless(-kx, ky)
    M2=model.M2
    Ham_p2=M2@Ham_p@M2.conj().T
    return (Ham_m-Ham_p2).__abs__().max()
def check_M2_spinful(model,kx,ky):
    Ham_p=model.get_Hamiltonian_monolayer_spinful(kx, ky)
    Ham_m=model.get_Hamiltonian_monolayer_spinful(-kx, ky)
    M2=np.kron(model.M2,2j*np.array([[0,1],[1,0]]))
    Ham_p2=M2@Ham_p@M2.conj().T
    return (Ham_m-Ham_p2).__abs__().max()
# TRS
def check_TRS_spinless(model,kx,ky):
    Ham_p=model.get_Hamiltonian_monolayer_spinless(kx, ky)
    Ham_m=model.get_Hamiltonian_monolayer_spinless(-kx, -ky)
    Ham_p2=Ham_p.conj()
    return (Ham_m-Ham_p2).__abs__().max()
def check_TRS_spinful(model,kx,ky):
    Ham_p=model.get_Hamiltonian_monolayer_spinful(kx, ky)
    Ham_m=model.get_Hamiltonian_monolayer_spinful(-kx, -ky)
    U=np.kron(np.eye(11),np.array([[0,-1],[1,0]]))
    Ham_p2=U@Ham_p.conj()@(U.conj().T)
    return (Ham_m-Ham_p2).__abs__().max()

# Bilayer 
def check_bilayer_M1_spinless(model,kx,ky,func=None):
    if func is None:
        func=model.get_Hamiltonian_spinless
    Ham_p=func(kx, ky,eff=1)
    M1=np.kron(np.fliplr(np.diag([1]*(len(model.theta_list)+1))),model.M1)
    Ham_p2=M1@Ham_p@M1.conj().T
    return (Ham_p-Ham_p2).__abs__().max()
def check_bilayer_M1_spinful(model,kx,ky,func=None):
    if func is None:
        func=model.get_Hamiltonian_spinful
    Ham_p=func(kx, ky,eff=1)
    M1=np.kron(np.kron(np.fliplr(np.diag([1]*(len(model.theta_list)+1))),model.M1),1j*np.array([[1,0],[0,-1]]))
    Ham_p2=M1@Ham_p@M1.conj().T
    return (Ham_p-Ham_p2).__abs__().max()

# C_2x= M1 * M3, z->-z, y-> -y
def check_bilayer_C2x_spinless(model,kx,ky,func=None):
    if func is None:
        func=model.get_Hamiltonian_spinless
    M=np.kron(np.fliplr(np.diag([1]*(len(model.theta_list)+1))),model.M1@model.M3)
    Ham_p=func(kx, ky,eff=1)
    Ham_m=func(kx, -ky,eff=1)
    Ham_m2=M@Ham_m@(M.conj().T)
    return (Ham_p-Ham_m2).__abs__().max()

def check_bilayer_C2x_spinful(model,kx,ky,func=None):
    if func is None:
        func=model.get_Hamiltonian_spinful
    M=np.kron(np.kron(np.fliplr(np.diag([1]*(len(model.theta_list)+1))),model.M1@model.M3),1j*np.array([[0,1],[1,0]]))
    Ham_p=model.get_Hamiltonian_spinful(kx, ky,eff=1)
    Ham_m=model.get_Hamiltonian_spinful(kx, -ky,eff=1)
    Ham_m2=M@Ham_m@(M.conj().T)
    return (Ham_p-Ham_m2).__abs__().max()

def check_bilayer_M2_spinless(model,kx,ky,func=None):
    if func is None:
        func=model.get_Hamiltonian_spinless
    Ham_p=func(kx, ky,eff=1)
    Ham_m=func(-kx, ky,eff=1)
    M2=np.kron(np.eye(len(model.theta_list)+1),model.M2)
    Ham_p2=M2@Ham_p@M2.conj().T
    return (Ham_m-Ham_p2).__abs__().max()
def check_bilayer_M2_spinful(model,kx,ky,func=None):
    if func is None:
        func=model.get_Hamiltonian_spinful
    Ham_p=func(kx, ky,eff=1)
    Ham_m=func(-kx, ky,eff=1)
    M2=np.kron(np.kron(np.eye(len(model.theta_list)+1),model.M2),1j*np.array([[0,1],[1,0]]))
    Ham_p2=M2@Ham_p@M2.conj().T
    return (Ham_m-Ham_p2).__abs__().max()

def check_bilayer_TRS_spinless(model,kx,ky,func=None):
    if func is None:
        func=model.get_Hamiltonian_spinless
    Ham_p=func(kx, ky,eff=1)
    Ham_m=func(-kx, -ky,eff=1)
    Ham_p2=Ham_p.conj()
    return (Ham_m-Ham_p2).__abs__().max()

def check_bilayer_TRS_spinful(model,kx,ky,func=None):
    if func is None:
        func=model.get_Hamiltonian_spinful
    Ham_p=func(kx, ky,eff=1)
    Ham_m=func(-kx, -ky,eff=1)
    U=np.kron(np.eye(Ham_p.shape[0]//2),np.array([[0,-1],[1,0]]))
    Ham_p2=U@Ham_p.conj()@(U.conj().T)
    return (Ham_m-Ham_p2).__abs__().max()
