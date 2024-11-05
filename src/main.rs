mod translator;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::ops::{Deref, Range};
use candle_core::{Module, DType, Device, IndexOp, Tensor};
use candle_nn::{embedding, Embedding, Optimizer, VarBuilder, VarMap};
use itertools::Itertools;
use rand::distributions::Distribution;
use rand::rngs::StdRng;
use rand::SeedableRng;
use crate::translator::{Translator, U32ToChar};

fn main() {
    // Read file and collect all unique tokens, sorted
    let mut file = BufReader::new(File::open(r"C:\Users\pokmo\OneDrive\Desktop\DevStuff\Rust\deep_learning\input.txt").expect("Should be a valid file"));
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("Should be able to read file");
    let unique_sorted = contents.as_bytes().iter().sorted().unique().collect_vec();
    let vocab_size = unique_sorted.len();

    let translator = Translator::new(unique_sorted.deref()).unwrap();

    let device = Device::cuda_if_available(0).expect("Should be able to leverage CUDA cores");

    let data = Tensor::new(contents.as_bytes(), &device).expect("Should be able to construct tensor");
    // print!("{}", data.i(..1000).unwrap());

    // Setup to use the first 90% for training data and the remaining 10% for validation
    let n = (0.9*data.dim(0).unwrap() as f32) as usize;
    let train_data = data.i(..n).unwrap();
    let validation_data = data.i(n..).unwrap();

    let batch_size = 4;
    let block_size = 8;

    // computationally expensive to train on the full dataset, so just take chunks from it.
    let (xb, yb) = get_batch(&train_data, batch_size, block_size);
    println!("{}", xb);
    println!("{}", yb);

    let var_map = VarMap::new();
    let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let mut model = BigramLanguageModel::new(vocab_size, var_builder);
    let (logits, loss) = ModuleWithLoss::forward(&model, &xb, &yb).unwrap();
    println!("{}", loss);
    let generated = model.generate(&Tensor::zeros((1, 1), DType::U32, &device).unwrap(), 100).unwrap();
    println!("{:?}", translator.u32_array_to_chars(&generated.to_vec2::<u32>().unwrap()[0]));

    let map_vars = var_map.all_vars();
    println!("{:?}", &map_vars);
    let mut optimizer = candle_nn::optim::AdamW::new_lr(map_vars, 1e-3).unwrap();

    let mut final_loss = None;
    for steps in 0..12000 {
        if steps % 300 == 0 {
            let (xn, yn) = get_batch(&train_data, 32, block_size);
            let (_, loss) = ModuleWithLoss::forward(&model, &xn, &yn).unwrap();
            println!("Loss: {}", loss);
        }
        let (xn, yn) = get_batch(&train_data, 32, block_size);

        let (_, loss) = ModuleWithLoss::forward(&model, &xn, &yn).unwrap();
        optimizer.backward_step(&loss).unwrap();
        final_loss = Some(loss);
    }
    println!("{}", final_loss.unwrap());

    let generated = model.generate(&Tensor::zeros((1, 1), DType::U32, &device).unwrap(), 100).unwrap();
    println!("{:?}", translator.u32_array_to_chars(&generated.to_vec2::<u32>().unwrap()[0]));
}

fn get_batch(tensor: &Tensor, batch_size: usize, block_size: usize) -> (Tensor, Tensor) {
    // generate random locations within data to use as block starting locations
    let ix = Tensor::rand(0.0, (tensor.dims1().unwrap() - block_size) as f32, (batch_size,), tensor.device())
        .and_then(|t| t.to_dtype(DType::U32)).unwrap();
    // builds the input values. Each set of these are associated with an index of the target
    // values, which serves as the "correct" answer to that set of inputs.
    let x = stack_range(tensor, ix.to_vec1().unwrap(), |i| i..i + block_size).unwrap();
    let y = stack_range(tensor, ix.to_vec1().unwrap(), |i| i + 1.. i + block_size + 1).unwrap();
    // println!("x: {:?}, y: {:?}", x.shape(), y.shape());
    (x, y)
}

fn stack_range<F: Fn(usize) -> Range<usize>>(tensor: &Tensor, indices: Vec<u32>, range: F) -> candle_core::Result<Tensor> {
    let mut vec = Vec::new();
    for i in indices {
        vec.push(tensor.i(range(i as usize))?);
    }
    Tensor::stack(vec.deref(), tensor.dims().len())
}

trait ModuleWithLoss {
    fn forward(&self, i: &Tensor, xs: &Tensor) -> candle_core::Result<(Tensor, Tensor)>;
}

struct BigramLanguageModel {
    token_embedding_table: Embedding,
    rng: StdRng
}

impl BigramLanguageModel {
    pub fn new(vocab_size: usize, var_builder: VarBuilder) -> Self {
        Self {
            // light wrapper over a tensor, meant for retrieving rows
            // NOTE: This cannot be of DType::I64 due to a lack of support for uexp_ for 64-bit signed integers in CUDA
            token_embedding_table: embedding(vocab_size, vocab_size, var_builder).unwrap(),
            rng: StdRng::seed_from_u64(1337u64),
        }
    }

    pub fn generate(&mut self, i: &Tensor, max_tokens: u32) -> candle_core::Result<Tensor> {
        let mut generated = i.clone();
        for _ in 0..max_tokens {
            let logits = generated.apply(self)?;
            let logits = logits.i((.., logits.dim(1)? - 1, ..))?;
            let probabilities = candle_nn::ops::softmax(&logits, 1)?;
            let distribution = rand::distributions::WeightedIndex::new(&probabilities.to_vec2::<f32>()?[0]).unwrap();
            let i_next = distribution.sample(&mut self.rng) as u32;
            let next_tensor = Tensor::full(i_next, 1, generated.device())?.unsqueeze(0)?;
            generated = Tensor::cat(&[generated, next_tensor], 1)?;
        }
        Ok(generated)
    }
}

impl Module for BigramLanguageModel {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        xs.apply(&self.token_embedding_table)
    }
}

impl ModuleWithLoss for BigramLanguageModel {
    fn forward(&self, i: &Tensor, xs: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
        // returns "logits", which allow for prediction from a single token, rather than a set of tokens
        let mut logits = i.apply(&self.token_embedding_table)?;
        let (b, t, c) = logits.dims3()?;
        let logits = logits.reshape((b*t, c))?;
        let targets = xs.reshape((b*t))?;
        // in this case, we expect loss to be -ln(1/65), so negative natlog of the inverse vocab size
        let loss = candle_nn::loss::cross_entropy(&logits, &targets)?;
        Ok((logits, loss))
    }
}