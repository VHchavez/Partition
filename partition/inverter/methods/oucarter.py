"""
oucarter.py

Functions associated with Ou-Carter inversion
"""

import numpy as np
from opt_einsum import contract
import psi4

class Oucarter():
    """
    Ou-Carter density to potential inversion [1].
    [1] [J. Chem. Theory Comput. 2018, 14, 5680−5689]
    """
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

    def _get_optimized_external_potential(self, grid_info, average_alpha_beta=False):
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
        Vpot = wfn_LDA.V_potential()

        vxc_LDA_DFT = self.grid.vxc(Da=Da_LDA, Db=Db_LDA, vpot=Vpot)
        vxc_LDA = self.grid.vxc(Da=Da_LDA, Db=Db_LDA, grid=grid_info)
        if self.ref != 1:
            assert vxc_LDA.shape[-1] == 2
            vxc_LDA_beta = vxc_LDA[:,1]
            vxc_LDA = vxc_LDA[:, 0]
            vxc_LDA_DFT_beta = vxc_LDA_DFT[:, 1]
            vxc_LDA_DFT = vxc_LDA_DFT[:, 0]

        print("About to calculate the components of vext!")

        print("Nalpha", Nalpha)

        # _average_local_orbital_energy() taken from mrks.py.
        e_bar_DFT = self._average_local_orbital_energy(Da_LDA, Ca_LDA[:, :Nalpha], epsilon_a_LDA[:Nalpha])
        e_bar     = self._average_local_orbital_energy(Da_LDA, Ca_LDA[:, :Nalpha], epsilon_a_LDA[:Nalpha], grid_info=grid_info)

        tauLmP_rho_DFT = self._get_l_kinetic_energy_density_directly(Da_LDA)
        tauLmP_rho     = self._get_l_kinetic_energy_density_directly(Da_LDA, grid_info=grid_info)

        tauP_rho_DFT = self._pauli_kinetic_energy_density(Da_LDA, Ca_LDA[:,:Nalpha])
        tauP_rho     = self._pauli_kinetic_energy_density(Da_LDA, Ca_LDA[:,:Nalpha], grid_info=grid_info)


        # # Grid INFO
        # self.ext_bar = e_bar
        # self.ext_taul = tauLmP_rho
        # self.ext_taup = tauP_rho

        # self.d = Da_LDA
        # self.c = Ca_LDA[:,:Nalpha]
        # self.e = epsilon_a_LDA[:Nalpha]

        tauL_rho_DFT = tauLmP_rho_DFT + tauP_rho_DFT
        tauL_rho     = tauLmP_rho + tauP_rho

        vext_opt_no_H_DFT = e_bar_DFT - tauL_rho_DFT - vxc_LDA_DFT
        vext_opt_no_H = e_bar - tauL_rho - vxc_LDA

        J = self.form_jk(Ca_LDA[:,:Nalpha],  Cb_LDA[:,:Nbeta])[0]
        vext_opt_no_H_DFT_Fock = self.grid.dft_grid_to_fock(vext_opt_no_H_DFT, Vpot)
        vext_opt_DFT_Fock = vext_opt_no_H_DFT_Fock - J[0] - J[1]
        vh_dft = self.grid.esp(Da=Da_LDA, Db=Db_LDA,vpot=Vpot)[1]
        vext_opt_dft = vext_opt_no_H_DFT - vh_dft
        vH = self.grid.esp(Da=Da_LDA, Db=Db_LDA,grid=grid_info)[1]
        vext_opt = vext_opt_no_H - vH
        # vext_opt -= shift

        

        if self.ref != 1:
            e_bar_DFT_beta = self._average_local_orbital_energy(Db_LDA, Cb_LDA[:,:Nbeta], epsilon_b_LDA[:Nbeta])
            e_bar_beta = self._average_local_orbital_energy(Db_LDA, Cb_LDA[:, :Nbeta], epsilon_b_LDA[:Nbeta], grid_info=grid_info)


            tauLmP_rho_DFT_beta = self._get_l_kinetic_energy_density_directly(Db_LDA, )
            tauLmP_rho_beta = self._get_l_kinetic_energy_density_directly(Db_LDA, grid_info=grid_info)

            tauP_rho_DFT_beta = self._pauli_kinetic_energy_density(Db_LDA, Cb_LDA[:,:Nbeta])
            tauP_rho_beta = self._pauli_kinetic_energy_density(Db_LDA, Cb_LDA[:,:Nbeta], grid_info=grid_info)

            tauL_rho_DFT_beta = tauLmP_rho_DFT_beta + tauP_rho_DFT_beta
            tauL_rho_beta = tauLmP_rho_beta + tauP_rho_beta

            vext_opt_no_H_DFT_beta = e_bar_DFT_beta - tauL_rho_DFT_beta - vxc_LDA_DFT_beta
            vext_opt_no_H_beta = e_bar_beta - tauL_rho_beta - vxc_LDA_beta

            vext_opt_no_H_DFT_Fock_beta = self.dft_grid_to_fock(vext_opt_no_H_DFT_beta, Vpot)
            vext_opt_DFT_Fock_beta = vext_opt_no_H_DFT_Fock_beta - J[0] - J[1]

            vext_opt_beta = vext_opt_no_H_beta - vH

            # vext_opt_DFT_Fock = (vext_opt_DFT_Fock + vext_opt_DFT_Fock_beta) * 0.5
            # vext_opt = (vext_opt + vext_opt_beta) * 0.5

            return (vext_opt_DFT_Fock, vext_opt_DFT_Fock_beta), (vext_opt, vext_opt_beta)
        return vext_opt_DFT_Fock, vext_opt, vext_opt_dft
    
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

            # print(l_phi.shape)
            # print(lC.shape)
            # print(eig.shape)
            # print(l_phi.shape)
            # print('Number of points', iw + l_npoints)

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

    def oucarter(self, maxiter, vxc_grid, D_tol=1e-7,
             eig_tol=1e-4, frac_old=0.5, init="scan"):

        self.Vpot = self.grid.vpot
        
        p4da = psi4.core.Matrix.from_array( self.dt[0] )
        p4db = psi4.core.Matrix.from_array( self.dt[1] )
        
        grid_info = self.grid.grid_to_blocks(vxc_grid)
        if self.ref == 1:
            grid_info[-1].set_pointers( p4da )
        else:
            grid_info[-1].set_pointers( p4da, p4db )

        print("About to calculate vext opt")
        if self.ref == 1:
            vext_opt_Fock, vext_opt, vext_opt_DFT = self._get_optimized_external_potential(grid_info)
        else:
            (vext_opt_Fock, vext_opt_Fock_beta), (vext_opt, vext_opt_beta) = self._get_optimized_external_potential(grid_info)

        # Make sure self.vb also has va
        vH0_Fock = self.va
        

        # Initialization.
        if init is None:
            self.Da = np.copy(self.Dt[0])
            self.Coca = np.copy(self.ct[0])
            self.eigvecs_a = self.wfn.epsilon_a().np[:Nalpha]

            self.Db = np.copy(self.Dt[1])
            self.Cocb = np.copy(self.ct[1])
            self.eigvecs_b = self.wfn.epsilon_b().np[:Nbeta]
        elif init.lower()=="continue":
            pass
        else:
            wfn_temp = psi4.energy(init+"/" + self.basis_str, 
                                   molecule=self.mol, 
                                   return_wfn=True)[1]
            Nalpha = wfn_temp.nalpha()
            Nbeta = wfn_temp.nbeta()
            self.nalpha = Nalpha
            self.nbeta  = Nbeta
            
            self.Da = np.array(wfn_temp.Da())
            self.Coca = np.array(wfn_temp.Ca())[:, :Nalpha]
            self.eigvecs_a = np.array(wfn_temp.epsilon_a())
            self.Db = np.array(wfn_temp.Db())
            self.Cocb = np.array(wfn_temp.Cb())[:, :Nbeta]
            self.eigvecs_b = np.array(wfn_temp.epsilon_b())
            del wfn_temp


        vxc_old = 0.0
        vxc_old_beta = 0.0
        Da_old = 0.0
        eig_old = 0.0
        tauLmP_rho = self._get_l_kinetic_energy_density_directly(self.dt[0])
        if self.ref != 1:
            tauLmP_rho_beta = self._get_l_kinetic_energy_density_directly(self.dt[1])

        for OC_step in range(1, maxiter+1):
            tauP_rho = self._pauli_kinetic_energy_density(self.Da, self.Coca)
            e_bar = self._average_local_orbital_energy(self.Da, self.Coca, self.eigvecs_a[:Nalpha])
            # shift = self.eigvecs_a[Nalpha - 1] - self.wfn.epsilon_a().np[Nalpha - 1]
            # vxc + vext_opt + vH0
            vxc_extH = e_bar - tauLmP_rho - tauP_rho #- shift # vxc+ext+h on the grid

            # DFT components
            self.dft_bar = e_bar.copy()
            self.dft_taul = tauLmP_rho.copy()
            self.dft_taup = tauP_rho.copy()

            if self.ref != 1:
                tauP_rho_beta = self._pauli_kinetic_energy_density(self.Db, self.Cocb)
                e_bar_beta = self._average_local_orbital_energy(self.Db, self.Cocb, self.eigvecs_b[:Nbeta])
                shift_beta = self.eigvecs_b[Nbeta - 1] - self.wfn.epsilon_b().np[Nbeta - 1]
                # vxc + vext_opt + vH0
                vxc_extH_beta = e_bar_beta - tauLmP_rho_beta - tauP_rho_beta #- shift_beta

            Derror = np.linalg.norm(self.Da - Da_old) / self.nbf ** 2
            eerror = (np.linalg.norm(self.eigvecs_a[:Nalpha] - eig_old) / Nalpha)
            if (Derror < D_tol) and (eerror < eig_tol):
                print("KSDFT stops updating.")
                break

            # linear Mixture
            if OC_step != 1:
                vxc_extH = vxc_extH * (1 - frac_old) + vxc_old * frac_old

                if self.ref != 1:
                    vxc_extH_beta = vxc_extH_beta * (1 - frac_old) + vxc_old_beta * frac_old

            vxc_old = np.copy(vxc_extH)
            if self.ref != 1:
                vxc_old_beta = np.copy(vxc_extH_beta)

            # Save old data.
            Da_old = np.copy(self.Da)
            eig_old = np.copy(self.eigvecs_a[:Nalpha])

            Vxc_extH_Fock = self.grid.dft_grid_to_fock(vxc_extH, self.Vpot)
            Vxc_Fock = Vxc_extH_Fock - vext_opt_Fock - vH0_Fock

            if self.ref != 1:
                Vxc_extH_Fock_beta = self.dft_grid_to_fock(vxc_extH_beta, self.Vpot)
                Vxc_Fock_beta = Vxc_extH_Fock_beta - vext_opt_Fock_beta - vH0_Fock

            if self.ref == 1:
                self._diagonalize_with_potential_vFock(v=Vxc_Fock)
            else:
                self._diagonalize_with_potential_vFock(v=(Vxc_Fock, Vxc_Fock_beta))


            print(f"Iter: {OC_step}, Density Change: {Derror:2.2e}, Eigenvalue Change: {eerror:2.2e}.")
            # nerror = self.on_grid_density(Da=self.Dt[0] - self.Da, Db=self.Dt[1] - self.Da, Vpot=self.Vpot)
            # nerror = np.sum(np.abs(nerror.T) * w)
            # print("nerror", nerror)

        # Calculate vxc on grid
        vH0 = self.grid.esp(Da=self.Da, Db=self.Db, grid=grid_info)[1]
        vH0_DFT = self.grid.esp(Da=self.Da, Db=self.Db,vpot=self.Vpot)[1]
        tauLmP_rho = self._get_l_kinetic_energy_density_directly(self.dt[0], grid_info=grid_info)
        tauP_rho = self._pauli_kinetic_energy_density(self.Da, self.Coca, grid_info=grid_info)
        # shift = self.eigvecs_a[Nalpha - 1] - self.wfn.epsilon_a().np[Nalpha - 1]
        e_bar = self._average_local_orbital_energy(self.Da, self.Coca, self.eigvecs_a[:Nalpha], grid_info=grid_info)
        if self.ref != 1:
            tauLmP_rho_beta = self._get_l_kinetic_energy_density_directly(self.dt[1], grid_info=grid_info)
            tauP_rho_beta = self._pauli_kinetic_energy_density(self.Db, self.Cocb, grid_info=grid_info)
            # shift_beta = self.eigvecs_b[Nbeta - 1] - self.wfn.epsilon_b().np[Nbeta - 1]
            e_bar_beta = self._average_local_orbital_energy(self.Db, self.Cocb, self.eigvecs_b[:Nbeta], grid_info=grid_info)

        if self.ref == 1:
            # self.grid.vxc = 
            # return self.grid.vxc, e_bar, tauLmP_rho, tauP_rho, vext_opt, vH0, shift

            vxc_dft = vxc_extH - vext_opt_DFT  - vH0_DFT

            #dft components
            self.dft_vh = vH0_DFT
            self.dft_vext = vext_opt_DFT



            #vxc plot components
            self.oc_vext = vext_opt
            self.oc_vh   = vH0
            self.oc_vxc = e_bar - tauLmP_rho - tauP_rho
            self.oc_e = e_bar
            self.oc_taul = tauLmP_rho
            self.oc_taup = tauP_rho

            vxc_plot = e_bar - tauLmP_rho - tauP_rho - vext_opt - vH0 #- shift

            return vxc_dft, vxc_plot

        else:
            self.grid.vxc = np.array((e_bar - tauLmP_rho - tauP_rho - vext_opt - vH0, #- shift,
                                      e_bar_beta - tauLmP_rho_beta - tauP_rho_beta - vext_opt_beta - vH0 #- shift_beta
                                      ))
            return self.grid.vxc, (e_bar, e_bar_beta), (tauLmP_rho, tauLmP_rho_beta), \
                   (tauP_rho,tauP_rho_beta), (vext_opt, vext_opt_beta), vH0, (shift, shift_beta)