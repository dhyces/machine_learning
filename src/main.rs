use candle_core::{Device, Tensor};

fn main() {
    let dev = Device::new_cuda(0).unwrap();

    let ten_0 = Tensor::randn(0f32, 1., (2,3), &dev).unwrap();
    let ten_1 = Tensor::randn(0f32, 1., (3,4), &dev).unwrap();

    let c = ten_0.matmul(&ten_1).unwrap();
    println!("{c:?}");
}