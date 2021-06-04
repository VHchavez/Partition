"""
oucarter.py
"""

import numpy as np
import psi4
from opt_einsum import contract

class Oucarter():
    """
    Ou-Carter density to potential inversion [1].
    [1] [J. Chem. Theory Comput. 2018, 14, 5680−5689]
    """

    def _get_l_kinetic_energy_density_directly(self, D, grid_info=None):
        """
        Calculate $\frac{\tau_L^{KS}}{\rho^{KS}}-\frac{\tau_P^{KS}}{\rho^{KS}}$:
        laplace_rho_temp: $\frac{\nabla^2 \rho}{4}$;
        tauW_temp: $\frac{|\napla \rho|^2}{8|\rho|}$;
        tauLmP_rho: $\frac{|\napla \rho|^2}{8|\rho|^2} - \frac{\nabla^2 \rho}{4\rho}$.
        (i.e. the 2dn and 3rd term in eqn. (17) in [1] over $\rho$.):
        """

        if grid_info is None:
            tauLmP_rho = np.zeros(self.Vpot.grid().npoints())
            nblocks = self.Vpot.nblocks()

            points_func = self.Vpot.properties()[0]
            blocks = None

        else:
            blocks, npoints, points_func = grid_info
            tauLmP_rho = np.zeros(npoints)
            nblocks = len(blocks)

        points_func.set_deriv(2)

        iw = 0
        for l_block in range(nblocks):
            # Obtain general grid information
            if blocks is None:
                l_grid = self.Vpot.get_block(l_block)
            else:
                l_grid = blocks[l_block]
            l_npoints = l_grid.npoints()

            points_func.compute_points(l_grid)
            l_lpos = np.array(l_grid.functions_local_to_global())
            if len(l_lpos) == 0:
                iw += l_npoints
                continue
            l_phi = np.array(points_func.basis_values()["PHI"])[:l_npoints, :l_lpos.shape[0]]
            l_phi_x = np.array(points_func.basis_values()["PHI_X"])[:l_npoints, :l_lpos.shape[0]]
            l_phi_y = np.array(points_func.basis_values()["PHI_Y"])[:l_npoints, :l_lpos.shape[0]]
            l_phi_z = np.array(points_func.basis_values()["PHI_Z"])[:l_npoints, :l_lpos.shape[0]]
            l_phi_xx = np.array(points_func.basis_values()["PHI_XX"])[:l_npoints, :l_lpos.shape[0]]
            l_phi_yy = np.array(points_func.basis_values()["PHI_YY"])[:l_npoints, :l_lpos.shape[0]]
            l_phi_zz = np.array(points_func.basis_values()["PHI_ZZ"])[:l_npoints, :l_lpos.shape[0]]

            lD = D[(l_lpos[:, None], l_lpos)]
            # lC = C[l_lpos, :]

            rho = contract('pm,mn,pn->p', l_phi, lD, l_phi)
            rho_inv = 1/rho

            # Calculate the second term
            laplace_rho_temp = contract('ab,pa,pb->p', lD, l_phi, l_phi_xx + l_phi_yy + l_phi_zz)
            # laplace_rho_temp += contract('pm, mn, pn->p', l_phi_x,lD, l_phi_x)
            # laplace_rho_temp += contract('pm, mn, pn->p', l_phi_y,lD, l_phi_y)
            # laplace_rho_temp += contract('pm, mn, pn->p', l_phi_z,lD, l_phi_z)
            laplace_rho_temp += np.sum((l_phi_x @ lD) * l_phi_x, axis=1)
            laplace_rho_temp += np.sum((l_phi_y @ lD) * l_phi_y, axis=1)
            laplace_rho_temp += np.sum((l_phi_z @ lD) * l_phi_z, axis=1)

            laplace_rho_temp *= 0.25 * 2

            # Calculate the third term
            tauW_temp = contract('pm, mn, pn->p', l_phi, lD, l_phi_x) ** 2
            tauW_temp += contract('pm, mn, pn->p', l_phi, lD, l_phi_y) ** 2
            tauW_temp += contract('pm, mn, pn->p', l_phi, lD, l_phi_z) ** 2
            tauW_temp *= rho_inv * 0.125 * 4

            tauLmP_rho[iw: iw + l_npoints] = (-laplace_rho_temp + tauW_temp) * rho_inv
            iw += l_npoints
        assert iw == tauLmP_rho.shape[0], "Somehow the whole space is not fully integrated."

        return tauLmP_rho

    def _get_optimized_external_potential(self, grid_info, average_alpha_beta=False, return_dft_grid=False):
        """
        $
        v^{~}{ext}(r) = \epsilon^{-LDA}(r)
        - \frac{\tau^{LDA}{L}}{n^{LDA}(r)}
        - v_{H}^{LDA}(r) - v_{xc}^{LDA}(r)
        $
        (22) in [1].
        """

        Nalpha = self.nalpha
        Nbeta = self.nbeta

        # SVWN calculation
        wfn_LDA = psi4.energy("SVWN/" + self.basis_str, molecule=self.mol, return_wfn=True)[1]
        Da_LDA = wfn_LDA.Da().np
        Db_LDA = wfn_LDA.Db().np
        Ca_LDA = wfn_LDA.Ca().np
        Cb_LDA = wfn_LDA.Cb().np
        epsilon_a_LDA = wfn_LDA.epsilon_a().np
        epsilon_b_LDA = wfn_LDA.epsilon_b().np
        # Vpot = wfn_LDA.V_potential()

        vxc_LDA_DFT = self.grid.vxc(Da=Da_LDA, Db=Db_LDA, vpot=self.grid.vpot)
        vxc_LDA = self.grid.vxc(Da=Da_LDA, Db=Db_LDA, grid=grid_info)

        if self.ref != 1:
            assert vxc_LDA.shape[-1] == 2
            vxc_LDA_beta = vxc_LDA[:,1]
            vxc_LDA = vxc_LDA[:, 0]
            vxc_LDA_DFT_beta = vxc_LDA_DFT[:, 1]
            vxc_LDA_DFT = vxc_LDA_DFT[:, 0]

        # _average_local_orbital_energy() taken from mrks.py.

        # Compute everything for ALPHA
        e_bar_DFT = self._average_local_orbital_energy(Da_LDA, Ca_LDA[:,:Nalpha], epsilon_a_LDA[:Nalpha])
        e_bar     = self._average_local_orbital_energy(Da_LDA, Ca_LDA[:, :Nalpha], epsilon_a_LDA[:Nalpha], grid_info=grid_info)

        tauLmP_rho_DFT = self._get_l_kinetic_energy_density_directly(Da_LDA)
        tauLmP_rho = self._get_l_kinetic_energy_density_directly(Da_LDA, grid_info=grid_info)

        tauP_rho_DFT = self._pauli_kinetic_energy_density(Da_LDA, Ca_LDA[:,:Nalpha])
        tauP_rho = self._pauli_kinetic_energy_density(Da_LDA, Ca_LDA[:,:Nalpha], grid_info=grid_info)

        tauL_rho_DFT = tauLmP_rho_DFT + tauP_rho_DFT
        tauL_rho = tauLmP_rho + tauP_rho

        vext_opt_no_H_DFT = e_bar_DFT - tauL_rho_DFT - vxc_LDA_DFT
        vext_opt_no_H = e_bar - tauL_rho - vxc_LDA

        J = self.form_jk(Ca_LDA[:,:Nalpha],  Cb_LDA[:,:Nbeta])[0]
        vext_opt_no_H_DFT_Fock = self.grid.dft_grid_to_fock(vext_opt_no_H_DFT, self.grid.vpot)
        vext_opt_DFT_Fock = vext_opt_no_H_DFT_Fock - J[0] - J[1]
        
        _, vH     = self.grid.esp(grid=grid_info, Da=wfn_LDA.Da().np, Db=wfn_LDA.Db().np, compute_hartree=True)
        _, vH_DFT = self.grid.esp(Da=wfn_LDA.Da().np, Db=wfn_LDA.Db().np, vpot=self.grid.vpot, compute_hartree=True)
        vext_opt = vext_opt_no_H - vH
        vext_DFT = vext_opt_no_H_DFT - vH_DFT
        # vext_opt -= shift

        if self.ref != 1:

            # Compute everything for BETA
            e_bar_DFT_beta = self._average_local_orbital_energy(Db_LDA, Cb_LDA[:,:Nbeta], epsilon_b_LDA[:Nbeta])
            e_bar_beta = self._average_local_orbital_energy(Db_LDA, Cb_LDA[:, :Nbeta], epsilon_b_LDA[:Nbeta], grid_info=grid_info)


            tauLmP_rho_DFT_beta = self._get_l_kinetic_energy_density_directly(Db_LDA)
            tauLmP_rho_beta = self._get_l_kinetic_energy_density_directly(Db_LDA, grid_info=grid_info)

            tauP_rho_DFT_beta = self._pauli_kinetic_energy_density(Db_LDA, Cb_LDA[:,:Nbeta])
            tauP_rho_beta = self._pauli_kinetic_energy_density(Db_LDA, Cb_LDA[:,:Nbeta], grid_info=grid_info)

            tauL_rho_DFT_beta = tauLmP_rho_DFT_beta + tauP_rho_DFT_beta
            tauL_rho_beta = tauLmP_rho_beta + tauP_rho_beta

            vext_opt_no_H_DFT_beta = e_bar_DFT_beta - tauL_rho_DFT_beta - vxc_LDA_DFT_beta
            vext_opt_no_H_beta = e_bar_beta - tauL_rho_beta - vxc_LDA_beta

            vext_opt_no_H_DFT_Fock_beta = self.grid.dft_grid_to_fock(vext_opt_no_H_DFT_beta, self.grid.vpot)
            vext_opt_DFT_Fock_beta = vext_opt_no_H_DFT_Fock_beta - J[0] - J[1]

            _, vH_beta     = self.grid.esp(grid=grid_info, Da=wfn_LDA.Da().np, Db=wfn_LDA.Db().np, compute_hartree=True)
            _, vH_DFT_beta = self.grid.esp(Da=wfn_LDA.Da().np, Db=wfn_LDA.Db().np, vpot=self.grid.vpot, compute_hartree=True)
            vext_opt_beta = vext_opt_no_H_beta - vH_beta
            vext_DFT_beta = vext_opt_no_H_DFT_beta - vH_DFT_beta

            if return_dft_grid:
                return vext_opt_DFT_Fock, vext_opt, vext_DFT, vext_opt_DFT_Fock_beta, vext_opt_beta, vext_DFT_beta 
            else:
                return (vext_opt_DFT_Fock, vext_opt_DFT_Fock_beta), (vext_opt, vext_opt_beta)
        
        if return_dft_grid:
            return vext_opt_DFT_Fock, vext_opt, vext_DFT
        else:
            return vext_opt_DFT_Fock, vext_opt
    
    def _average_local_orbital_energy(self, D, C, eig, grid_info=None):
        """
        (4)(6) in mRKS.
        """

        # Nalpha = self.molecule.nalpha
        # Nbeta = self.molecule.nbeta

        if grid_info is None:
            e_bar = np.zeros(self.Vpot.grid().npoints())
            nblocks = self.Vpot.nblocks()

            points_func = self.Vpot.properties()[0]
            points_func.set_deriv(0)
            blocks = None
        else:
            blocks, npoints, points_func = grid_info
            e_bar = np.zeros(npoints)
            nblocks = len(blocks)

            points_func.set_deriv(0)

        # For unrestricted
        iw = 0
        for l_block in range(nblocks):
            # Obtain general grid information
            if blocks is None:
                l_grid = self.Vpot.get_block(l_block)
            else:
                l_grid = blocks[l_block]
            l_npoints = l_grid.npoints()

            points_func.compute_points(l_grid)
            l_lpos = np.array(l_grid.functions_local_to_global())
            if len(l_lpos) == 0:
                iw += l_npoints
                continue
            l_phi = np.array(points_func.basis_values()["PHI"])[:l_npoints, :l_lpos.shape[0]]
            lD = D[(l_lpos[:, None], l_lpos)]
            lC = C[l_lpos, :]
            rho = contract('pm,mn,pn->p', l_phi, lD, l_phi)
            e_bar[iw:iw + l_npoints] = contract("pm,mi,ni,i,pn->p", l_phi, lC, lC, eig, l_phi) / rho

            iw += l_npoints
        assert iw == e_bar.shape[0], "Somehow the whole space is not fully integrated."
        return e_bar

    def _pauli_kinetic_energy_density(self, D, C, occ=None, grid_info=None):
        """
        (16)(18) in mRKS. But notice this does not return taup but taup/n
        :return:
        """

        if occ is None:
            occ = np.ones(C.shape[1])

        if grid_info is None:
            taup_rho = np.zeros(self.Vpot.grid().npoints())
            nblocks = self.Vpot.nblocks()

            points_func = self.Vpot.properties()[0]
            points_func.set_deriv(1)
            blocks = None

        else:
            blocks, npoints, points_func = grid_info
            taup_rho = np.zeros(npoints)
            nblocks = len(blocks)

            points_func.set_deriv(1)

        iw = 0
        for l_block in range(nblocks):
            # Obtain general grid information
            if blocks is None:
                l_grid = self.Vpot.get_block(l_block)
            else:
                l_grid = blocks[l_block]
            l_npoints = l_grid.npoints()

            points_func.compute_points(l_grid)
            l_lpos = np.array(l_grid.functions_local_to_global())
            if len(l_lpos) == 0:
                iw += l_npoints
                continue
            l_phi = np.array(points_func.basis_values()["PHI"])[:l_npoints, :l_lpos.shape[0]]
            l_phi_x = np.array(points_func.basis_values()["PHI_X"])[:l_npoints, :l_lpos.shape[0]]
            l_phi_y = np.array(points_func.basis_values()["PHI_Y"])[:l_npoints, :l_lpos.shape[0]]
            l_phi_z = np.array(points_func.basis_values()["PHI_Z"])[:l_npoints, :l_lpos.shape[0]]

            lD = D[(l_lpos[:, None], l_lpos)]

            rho = contract('pm,mn,pn->p', l_phi, lD, l_phi)

            lC = C[l_lpos, :]
            # Matrix Methods
            part_x = contract('pm,mi,nj,pn->ijp', l_phi, lC, lC, l_phi_x)
            part_y = contract('pm,mi,nj,pn->ijp', l_phi, lC, lC, l_phi_y)
            part_z = contract('pm,mi,nj,pn->ijp', l_phi, lC, lC, l_phi_z)
            part1_x = (part_x - np.transpose(part_x, (1, 0, 2))) ** 2
            part1_y = (part_y - np.transpose(part_y, (1, 0, 2))) ** 2
            part1_z = (part_z - np.transpose(part_z, (1, 0, 2))) ** 2


            occ_matrix = np.expand_dims(occ, axis=1) @ np.expand_dims(occ, axis=0)

            taup = np.sum((part1_x + part1_y + part1_z).T * occ_matrix, axis=(1,2)) * 0.5

            taup_rho[iw:iw + l_npoints] = taup / rho ** 2 * 0.5

            iw += l_npoints
        assert iw == taup_rho.shape[0], "Somehow the whole space is not fully integrated."
        return taup_rho

    def _diagonalize_with_potential_vFock(self, v=None):
        """
        Diagonalize Fock matrix with additional external potential
        """

        if v is None:
            fock_a = self.V + self.T + self.va
        else:
            if self.ref == 1:
                fock_a = self.V + self.T + self.va + v
            else:
                valpha, vbeta = v
                fock_a = self.V + self.T + self.va + valpha
                fock_b = self.V + self.T + self.vb + vbeta



        self.Ca, self.Coca, self.Da, self.eigvecs_a = self.diagonalize( fock_a, self.nalpha )

        if self.ref == 1:
            self.Cb, self.Cocb, self.Db, self.eigvecs_b = self.Ca.copy(), self.Coca.copy(), self.Da.copy(), self.eigvecs_a.copy()
        else:
            self.Cb, self.Cocb, self.Db, self.eigvecs_b = self.diagonalize( fock_b, self.nbeta )

    def oucarter(self, maxiter, vxc_grid, D_tol=6e-7,
             eig_tol=3e-7, frac_old=0.9 , init="scan"):
        """
        (23) in [1].
        [1] [J. Chem. Theory Comput. 2018, 14, 5680−5689]
        parameters:
        ----------------------
            maxiter: int
                same as opt_max_iter
            vxc_grid: np.ndarray of shape (3, num_grid_points)
                The final result will be represented on this grid
                default: 1e-4
            D_tol: float, opt
                convergence criteria for density matrices.
                default: 1e-7
            eig_tol: float, opt
                convergence criteria for occupied eigenvalue spectrum.
                default: 1e-4
            frac_old: float, opt
                Linear mixing parameter for current vxc and old vxc.
                If 0, no old vxc is mixed in.
                Should be in [0,1)
                default: 0.5.
            init: string, opt
                Initial guess method.
                default: "SCAN"
                1) If None, input wfn info will be used as initial guess.
                2) If "continue" is given, then it will not initialize
                but use the densities and orbitals stored. Meaningly,
                one can run a quick WY calculation as the initial
                guess. This can also be used to user speficified
                initial guess by setting Da, Coca, eigvec_a.
                3) If it's not continue, it would be expecting a
                method name string that works for psi4. A separate psi4 calculation
                would be performed."""

    # Nalpha = self.nalpha
    # Nbeta  = self.nbeta
        self.Vpot = self.grid.vpot

        # Generate jk object
        # self.jk = self.generate_jk()

        # Calculate hartree as the guide potential
        _, self.va0 = self.grid.esp(Da=self.dt[0], Db=self.dt[1], vpot=self.Vpot, compute_hartree=True)
        self.va     = self.grid.dft_grid_to_fock( self.va0, self.grid.vpot )
        self.vb     = self.va
        vH0_Fock    = self.va 

        # Set pointers for somthing?
        p4da = psi4.core.Matrix.from_array(self.dt[0])
        p4db = psi4.core.Matrix.from_array(self.dt[1])
        grid_info = self.grid.grid_to_blocks(vxc_grid)
        if self.ref == 1:
            grid_info[-1].set_pointers(p4da)
        else:
            grid_info[-1].set_pointers(p4da, p4db)

        # Initialization. 
        if init is None:
            raise ValueError("ARGH")
            # self.Da = np.copy(self.nt[0])
            # self.Coca = np.copy(self.ct[0])
            # self.eigvecs_a = self.wfn.epsilon_a().np[:Nalpha]

            # self.Db = np.copy(self.nt[1])
            # self.Cocb = np.copy(self.ct[1])
            # self.eigvecs_b = self.wfn.epsilon_b().np[:Nbeta]
        elif init.lower()=="continue":
            print("ARGH")
            pass
        else:
            # Initial SVWN for External Potential
            wfn_temp = psi4.energy(init+"/" + self.basis_str, 
                                   molecule=self.mol, 
                                   return_wfn=True)[1]
            self.nalpha = wfn_temp.nalpha()
            self.nbeta  = wfn_temp.nbeta()
            Nalpha = self.nalpha
            Nbeta = self.nbeta
            self.Da = np.array(wfn_temp.Da())
            self.Coca = np.array(wfn_temp.Ca())[:, :Nalpha]
            self.eigvecs_a = np.array(wfn_temp.epsilon_a())
            self.Db = np.array(wfn_temp.Db())
            self.Cocb = np.array(wfn_temp.Cb())[:, :Nbeta]
            self.eigvecs_b = np.array(wfn_temp.epsilon_b())

            del wfn_temp

        # Get effective external potential
        if self.ref == 1:
            vext_opt_Fock, vext_opt, vext_DFT = self._get_optimized_external_potential(grid_info, return_dft_grid=True)
        else:
            # (vext_opt_Fock, vext_opt_Fock_beta), (vext_opt, vext_opt_beta) = self._get_optimized_external_potential(grid_info)
            vext_opt_Fock, vext_opt, vext_DFT, vext_opt_Fock_beta, vext_opt_beta, vext_DFT_beta = self._get_optimized_external_potential(grid_info, return_dft_grid=True)


        vxc_old = 0.0
        vxc_old_beta = 0.0
        Da_old = 0.0
        eig_old = 0.0
        tauLmP_rho = self._get_l_kinetic_energy_density_directly(self.dt[0])
        if self.ref != 1:
            tauLmP_rho_beta = self._get_l_kinetic_energy_density_directly(self.dt[1])

        # ----> Begin SCF:
        for OC_step in range(1, maxiter+1):

            # ALPHA components
            tauP_rho = self._pauli_kinetic_energy_density(self.Da, self.Coca)
            e_bar    = self._average_local_orbital_energy(self.Da, self.Coca, self.eigvecs_a[:Nalpha])
            #shift    = self.eigvecs_a[Nalpha - 1] - self.wfn.epsilon_a().np[Nalpha - 1]
            vxc_extH = e_bar - tauLmP_rho - tauP_rho #- shift

                # BETA components
            if self.ref != 1:
                tauP_rho_beta = self._pauli_kinetic_energy_density(self.Db, self.Cocb)
                e_bar_beta = self._average_local_orbital_energy(self.Db, self.Cocb, self.eigvecs_b[:Nbeta])
                # shift_beta = self.eigvecs_b[Nbeta - 1] - self.wfn.epsilon_b().np[Nbeta - 1]
                vxc_extH_beta = e_bar_beta - tauLmP_rho_beta - tauP_rho_beta #- shift_beta

            Derror = np.linalg.norm(self.Da - Da_old) #/ self.nbf ** 2
            eerror = (np.linalg.norm(self.eigvecs_a[:Nalpha] - eig_old) / Nalpha)
            if (Derror < D_tol) and (eerror < eig_tol):
                print("KSDFT stops updating.")
                break

            if OC_step != 1:
                vxc_extH = vxc_extH * (1 - frac_old) + vxc_old * frac_old
                vxc_old = np.copy(vxc_extH)

                if self.ref != 1:
                    vxc_extH_beta = vxc_extH_beta * (1 - frac_old) + vxc_old_beta * frac_old
                    vxc_old_beta = np.copy(vxc_extH_beta)

                    Vxc_extH_Fock_beta = self.grid.dft_grid_to_fock(vxc_extH_beta, self.Vpot)
                    Vxc_Fock_beta = Vxc_extH_Fock_beta - vext_opt_Fock - vH0_Fock

            # Save old data.
            Da_old = np.copy(self.Da)
            eig_old = np.copy(self.eigvecs_a[:Nalpha])

            Vxc_extH_Fock = self.grid.dft_grid_to_fock(vxc_extH, self.Vpot)
            Vxc_Fock      = Vxc_extH_Fock - vext_opt_Fock - vH0_Fock
            vxc_dft       = vxc_extH      - vext_DFT      - self.va0

            # Compute vxc on basis
            Vxc_extH_Fock = self.grid.dft_grid_to_fock(vxc_extH, self.Vpot)
            Vxc_Fock      = Vxc_extH_Fock - vext_opt_Fock - vH0_Fock
            vxc_dft       = vxc_extH      - vext_DFT      - self.va0
                
            if self.ref == 1:
                self._diagonalize_with_potential_vFock(v=Vxc_Fock)
            else:
                self._diagonalize_with_potential_vFock(v=(Vxc_Fock, Vxc_Fock_beta))


            print(f"\t\t\t\t\tIter: {OC_step}, Density Change: {Derror:2.2e}, Eigenvalue Change: {eerror:2.2e}.")
            # nerror = self.on_grid_density(Da=self.nt[0] - self.Da, Db=self.nt[1] - self.Da, Vpot=self.Vpot)
            # nerror = np.sum(np.abs(nerror.T) * w)
            # print("nerror", nerror)

        # Get things on GRID
        vH0 = self.grid.esp(Da=self.dt[0], Db=self.dt[1], grid=grid_info, compute_hartree=True)[1]
        tauLmP_rho = self._get_l_kinetic_energy_density_directly(self.dt[0], grid_info=grid_info)
        tauP_rho = self._pauli_kinetic_energy_density(self.Da, self.Coca, grid_info=grid_info)
        e_bar = self._average_local_orbital_energy(self.Da, self.Coca, self.eigvecs_a[:Nalpha], grid_info=grid_info)
        # shift = self.eigvecs_a[Nalpha - 1] - self.wfn.epsilon_a().np[Nalpha - 1]
        if self.ref != 1:
            tauLmP_rho_beta = self._get_l_kinetic_energy_density_directly(self.dt[1], grid_info=grid_info)
            tauP_rho_beta = self._pauli_kinetic_energy_density(self.Db, self.Cocb, grid_info=grid_info)
            e_bar_beta = self._average_local_orbital_energy(self.Db, self.Cocb, self.eigvecs_b[:Nbeta], grid_info=grid_info)
            # shift_beta = self.eigvecs_b[Nbeta - 1] - self.wfn.epsilon_b().np[Nbeta - 1]

        # PLOT STUFF
        if self.ref == 1:
            #vxc_extH = e_bar - tauLmP_rho - tauP_rho
            vxc_plotter = e_bar - tauLmP_rho - tauP_rho - vext_opt - vH0   #- shift

            self.Plotter.oc_vxca = e_bar - tauLmP_rho - tauP_rho
            self.Plotter.oc_tauLa = tauLmP_rho
            self.Plotter.oc_tauPa = tauP_rho
            self.Plotter.oc_vext = vext_opt
            self.Plotter.oc_vh      = vH0
            
            #Tried different shit
            # vxc_dft = vxc_extH
            # vxc_plotter = e_bar - tauLmP_rho - tauP_rho
            # return vxc, e_bar, tauLmP_rho, tauP_rho, vext_opt, vH0 #, shift
        else:

            self.Plotter.oc_vxca = e_bar - tauLmP_rho - tauP_rho
            self.Plotter.oc_vxcb = e_bar_beta - tauLmP_rho_beta - tauP_rho_beta

            self.Plotter.oc_tauLa = tauLmP_rho
            self.Plotter.oc_tauLb = tauLmP_rho_beta

            self.Plotter.oc_tauPa = tauP_rho
            self.Plotter.oc_tauPb = tauP_rho_beta

            self.Plotter.oc_vext = vext_opt
            self.Plotter.oc_vh      = vH0

            

            vxc_dft_a = vxc_extH      - vext_DFT - self.va0
            vxc_dft_b = vxc_extH_beta - vext_DFT - self.va0
            vxca_plotter = e_bar - tauLmP_rho - tauP_rho                - vext_opt - vH0
            vxcb_plotter = e_bar_beta - tauLmP_rho_beta - tauP_rho_beta - vext_opt - vH0

            vxc_dft     = np.concatenate( (vxc_dft_a[None,:], vxc_dft_b[None,:]) )
            vxc_plotter = np.concatenate( (vxca_plotter[None,:], vxcb_plotter[None,:]) )


        return vxc_dft, vxc_plotter

            # vxca = e_bar - tauLmP_rho - tauP_rho
            # vxcb =
            # vxc = np.array((e_bar - tauLmP_rho - tauP_rho, #- vext_opt - vH0, #- shift,
            #                           e_bar_beta - tauLmP_rho_beta - tauP_rho_beta #- vext_opt_beta - vH0,# - shift_beta
            #                           ))
            # return vxc, (e_bar, e_bar_beta), (tauLmP_rho, tauLmP_rho_beta), \
            #        (tauP_rho,tauP_rho_beta), (vext_opt, vext_opt_beta), vH0, #(shift, shift_beta)