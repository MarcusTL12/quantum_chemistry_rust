pub mod hf;

pub use gaussian_basis::*;

#[cfg(test)]
mod tests {
    use crate::*;

    use clap::Parser;

    #[derive(Parser, Debug)]
    struct Args {
        #[clap(long)]
        test_threads: Option<usize>,

        #[clap(long, parse(from_flag))]
        show_output: bool,

        #[clap(long, parse(from_flag))]
        nocapture: bool,
    }

    fn get_num_test_jobs() -> usize {
        let args = Args::parse();

        args.test_threads.unwrap_or(num_cpus::get())
    }

    fn set_threads() {
        let test_threads = get_num_test_jobs();

        let threads = num_cpus::get() / test_threads;

        println!("Setting threads to {}", threads);

        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap_or(());
    }

    #[test]
    fn test_rhf() {
        set_threads();

        let mol = Molecule::new(
            parse_atoms(
                "
    O   0.0     0.0     0.0
    H   1.0     0.0     0.0
    H   0.0     1.0     0.0
",
            ),
            &mut get_basis("sto-3g"),
        );

        let mut rhf = hf::RHF::new(mol);

        let e_hf = rhf.run();

        println!("Final energy: {}", e_hf);
    }
}
