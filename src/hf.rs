use gaussian_basis::Molecule;

use ndarray::prelude::*;

pub struct RHF {
    mol: Molecule,
    ao_fock: Option<Array2<f64>>,
    ao_density: Option<Array2<f64>>,
    e_hf: Option<f64>,
    n_occ: usize,
    n_vir: usize,
}

impl RHF {
    pub fn new(mol: Molecule) -> Self {
        if mol.get_n_el() % 2 != 0 {
            panic!("RHF does not support an odd number of electrons!");
        }

        let n_occ = mol.get_n_el() / 2;

        Self {
            ao_fock: None,
            ao_density: None,
            n_occ,
            n_vir: mol.get_n_ao() - n_occ,
            mol,
            e_hf: None,
        }
    }

    fn update_fock_and_energy(&mut self) {
        let ao_fock = if let Some(mut ao_fock) = self.ao_fock.take() {
            ao_fock.fill(0.0);
            Some(ao_fock)
        } else {
            None
        };

        let ao_density = self
            .ao_density
            .take()
            .expect("Cannot construct fock matrix without density");

        let ao_fock = self.mol.construct_ao_fock(ao_density.view(), ao_fock);

        let fock_vec = ao_fock
            .view()
            .into_shape((self.mol.get_n_ao().pow(2),))
            .unwrap();

        let dens_vec = ao_density
            .view()
            .into_shape((self.mol.get_n_ao().pow(2),))
            .unwrap();

        self.e_hf = Some(fock_vec.dot(&dens_vec));

        self.ao_fock = Some(ao_fock);
        self.ao_density = Some(ao_density);
    }
}
