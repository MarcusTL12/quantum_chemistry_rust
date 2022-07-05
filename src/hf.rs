use gaussian_basis::*;

use ndarray_linalg::*;

const MAX_ITERS: usize = 100;
const GRAD_THRESHOLD: f64 = 1e-10;

pub struct RHF {
    mol: Molecule,
    ao_overlap: Option<Array2<f64>>,
    ao_fock: Option<Array2<f64>>,
    ao_density: Option<Array2<f64>>,
    mo_coeff: Option<Array2<f64>>,
    mo_energies: Option<Vec<f64>>,
    buffers: Vec<Array2<f64>>,
    e_1e: Option<f64>,
    e_hf: Option<f64>,
    n_occ: usize,
}

impl RHF {
    pub fn new(mol: Molecule) -> Self {
        if mol.get_n_el() % 2 != 0 {
            panic!("RHF does not support an odd number of electrons!");
        }

        let n_occ = mol.get_n_el() / 2;

        Self {
            ao_overlap: None,
            ao_fock: None,
            ao_density: None,
            mo_coeff: None,
            mo_energies: None,
            n_occ,
            mol,
            e_1e: None,
            e_hf: None,
            buffers: Vec::new(),
        }
    }

    pub fn run(&mut self) -> f64 {
        if self.ao_density.is_none() {
            self.set_sad_guess();
        }

        self.update_fock_and_energy();

        let mut last_e = self.e_hf.unwrap();

        println!("Initial energy: {}", last_e);

        for i in 0..MAX_ITERS {
            self.solve_roothan_hall_and_update_density();

            self.update_fock_and_energy();

            let maxgrad = self.get_max_grad();

            let e = self.e_hf.unwrap();
            let delta_e = e - last_e;
            last_e = e;

            println!(
                "i: {} E: {:.15}, max_grad: {:.2e}, ΔE: {:+.2e}",
                i, e, maxgrad, delta_e
            );

            if maxgrad < GRAD_THRESHOLD {
                break;
            }
        }

        self.e_hf.unwrap()
    }

    fn update_fock_and_energy(&mut self) {
        let prealloc = if let Some(mut ao_fock) = self.ao_fock.take() {
            ao_fock.fill(0.0);
            Some(ao_fock)
        } else {
            None
        };

        let ao_density = self
            .ao_density
            .take()
            .expect("Can't construct fock matrix without density!");

        let ao_h = self.mol.construct_ao_h(prealloc);

        let h_vec = ao_h
            .view()
            .into_shape((self.mol.get_n_ao().pow(2),))
            .unwrap();

        let dens_vec = ao_density
            .view()
            .into_shape((self.mol.get_n_ao().pow(2),))
            .unwrap();

        self.e_1e = Some(self.mol.get_nuc_rep() + 0.5 * h_vec.dot(&dens_vec));

        let ao_fock = self.mol.construct_ao_g(ao_density.view(), Some(ao_h));

        let fock_vec = ao_fock
            .view()
            .into_shape((self.mol.get_n_ao().pow(2),))
            .unwrap();

        let dens_vec = ao_density
            .view()
            .into_shape((self.mol.get_n_ao().pow(2),))
            .unwrap();

        self.e_hf = Some(self.e_1e.unwrap() + 0.5 * fock_vec.dot(&dens_vec));

        self.ao_fock = Some(ao_fock);
        self.ao_density = Some(ao_density);
    }

    fn set_sad_guess(&mut self) {
        let mut density = self.ao_density.take().unwrap_or(Array2::zeros((
            self.mol.get_n_ao(),
            self.mol.get_n_ao(),
        )));

        density.fill(0.0);

        for x in density.diag_mut().iter_mut().take(self.n_occ) {
            *x = 2.0;
        }

        self.ao_density = Some(density);
    }

    fn set_ao_overlap(&mut self, prealloc: Option<Array2<f64>>) {
        if self.ao_overlap.is_none() {
            let sh = (self.mol.get_n_ao(), self.mol.get_n_ao());

            let prealloc = prealloc.and_then(|mut p| {
                p.fill(0.0);
                Some(p.into_shape((sh.0, sh.1, 1)).unwrap())
            });

            self.ao_overlap = Some(
                self.mol
                    .construct_int1e_sym(cint1e!(int1e_ovlp_sph), 1, prealloc)
                    .into_shape(sh)
                    .unwrap(),
            );
        }
    }

    fn update_density(&mut self) {
        let mut density = self.ao_density.take().unwrap_or(Array2::zeros((
            self.mol.get_n_ao(),
            self.mol.get_n_ao(),
        )));

        let c = self
            .mo_coeff
            .take()
            .expect("Can't construct ao-density without mo-coefficients");

        let c_occ = c.slice(s![.., 0..self.n_occ]);

        linalg::general_mat_mul(1.0, &c_occ, &c_occ.t(), 0.0, &mut density);

        density *= 2.0;

        self.mo_coeff = Some(c);

        self.ao_density = Some(density);
    }

    fn solve_roothan_hall_and_update_density(&mut self) {
        self.set_ao_overlap(None);
        let ovlp = self.ao_overlap.take().unwrap();

        let fock = self
            .ao_fock
            .take()
            .expect("Can't solve roothan-hall eq. without fock matrix!");

        self.ao_fock = Some(if let Some(mut c) = self.mo_coeff.take() {
            c.assign(&fock);
            c
        } else {
            fock.clone()
        });

        let mut fock_ovlp = (fock, ovlp);
        let (e, _) = fock_ovlp.eigh_inplace(UPLO::Upper).unwrap();

        self.mo_energies = Some(e.into_raw_vec());

        let (c, s) = fock_ovlp;
        self.set_ao_overlap(Some(s));

        self.mo_coeff = Some(c);
        self.update_density();
    }

    fn get_buffer(&mut self) -> Array2<f64> {
        self.buffers.pop().unwrap_or(Array2::zeros((
            self.mol.get_n_ao(),
            self.mol.get_n_ao(),
        )))
    }

    fn return_buffer(&mut self, buf: Array2<f64>) {
        self.buffers.push(buf);
    }

    fn get_max_grad(&mut self) -> f64 {
        let mut f_c = self.get_buffer();
        let mut c_f_c = self.get_buffer();

        let c = self
            .mo_coeff
            .as_ref()
            .expect("Can't compute gradient without MO coefficients");

        linalg::general_mat_mul(
            1.0,
            self.ao_fock
                .as_ref()
                .expect("Can't compute gradient without Fock matrix"),
            c,
            0.0,
            &mut f_c,
        );

        linalg::general_mat_mul(1.0, &c.t(), &f_c, 0.0, &mut c_f_c);

        let ans = c_f_c
            .slice(s![0..self.n_occ, self.n_occ..self.mol.get_n_ao()])
            .iter()
            .map(|x| x.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        self.return_buffer(c_f_c);
        self.return_buffer(f_c);

        ans
    }
}
