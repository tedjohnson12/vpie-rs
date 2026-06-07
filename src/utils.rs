//! Utilities that are helpfull for VPIE
//! 

use core::f64;
use ndarray as nd;
use log;


pub fn bin_image(
    image: &nd::Array2<f64>,
    nwl: usize,
    ntime: usize,
    power: i32
) -> nd::Array2<f64>
{
    let original_size_time = image.nrows();
    let original_size_wl = image.ncols();
    let new_size_time = original_size_time.div_ceil(ntime);
    let new_size_wl = original_size_wl.div_ceil(nwl);
    let mut binned_image = nd::Array2::<f64>::zeros((new_size_time, new_size_wl));
    for i in 0..new_size_time {
        for j in 0..new_size_wl {
            let mut sum = 0.0;
            let mut count = 0;
            let time_window_start = i * ntime;
            let time_window_end = ((i + 1) * ntime).min(original_size_time);
            let wl_window_start = j * nwl;
            let wl_window_end = ((j + 1) * nwl).min(original_size_wl);
            for k in time_window_start..time_window_end {
                for l in wl_window_start..wl_window_end {
                    sum += image[[k, l]].powi(power);
                    count += 1;
                }
            }
            binned_image[[i, j]] = sum.powf(1.0 / power as f64) / count as f64;
        }
    }
    binned_image
}



#[cfg(test)]
mod test {

    use super::*;
    #[test]
    fn simple_test() {
        let image = nd::array![[1.0, 1.0], [1.0, 1.0]];
        let binned_image = bin_image(&image, 2, 2, 1);
        assert_eq!(binned_image, nd::array![[1.0]]);
    }
    #[test]
    fn uneven_test() {
        let image = nd::array![[1.0, 1.0,2.0], [1.0, 1.0,2.0]];
        let binned_image = bin_image(&image, 2, 2, 1);
        assert_eq!(binned_image, nd::array![[1.0,2.0]]);
    }
    #[test]
    fn oned_test() {
        let image = nd::array![[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]];
        let binned_image = bin_image(&image, 2, 1, 1);
        assert_eq!(binned_image, nd::array![[1.0,1.0,1.0]]);
    }
    #[test]
    fn inverse_test() {
        let image = nd::array![[3.0, 3.0, 1.0, 1.0]];
        let binned_image = bin_image(&image, 2, 1, -1);
        log::info!("{:?}", binned_image);
        assert_eq!(binned_image, nd::array![[1.5/2.0,0.25]]);
    }
    #[test]
    fn quadratic_test() {
        let image = nd::array![[1.0, 1.0, 1.0, 1.0]];
        let binned_image = bin_image(&image, 4, 1, 2);
        log::info!("{:?}", binned_image);
        assert_eq!(binned_image, nd::array![[0.5]]);
    }
}