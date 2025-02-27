mod translator;

use std::fs::File;
use std::io::{BufReader, Read};
use std::ops::{Deref};
use candle_core::{Module, DType, Device, IndexOp, Tensor};
use candle_nn::{embedding, Embedding, Optimizer, VarBuilder, VarMap};
use itertools::Itertools;
use rand::distributions::Distribution;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use crate::translator::{CharToU32, Translator, U32ToChar};

const RAND_SEED: u64 = 1337;
const TRAINING_ITERATIONS: u64 = 6000;
const EVAL_ITERATIONS: u64 = 300;
const LEARNING_RATE: f64 = 1e-3;

fn main() -> candle_core::Result<()> {
    // Read file and collect all unique tokens, sorted
    let mut file = BufReader::new(File::open(r".\input.txt").expect("Should be a valid file"));
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("Should be able to read file");
    let unique_sorted = contents.as_bytes().iter().sorted().unique().collect_vec();
    let vocab_size = unique_sorted.len();

    let translator = Translator::new(unique_sorted.deref()).unwrap();

    let device = Device::cuda_if_available(0).expect("Should be able to leverage CUDA cores");

    let data = translator.char_array_to_u32s(&contents.chars().collect::<Vec<char>>()).unwrap();
    // print!("{}", data.i(..1000).unwrap());

    // Setup to use the first 90% for training data and the remaining 10% for validation
    let n = (data.len() * 9) / 10;
    let (train_data, validation_data) = data.split_at(n);

    let mut rng = StdRng::seed_from_u64(RAND_SEED);

    let batch_size = 4;
    let block_size = 8;

    // computationally expensive to train on the full dataset, so just take chunks from it.
    let (xb, yb) = get_batch(&train_data, &device, batch_size, block_size, &mut rng)?;
    println!("{}", xb);
    println!("{}", yb);

    let var_map = VarMap::new();
    // NOTE: This cannot be of DType::I64 due to a lack of support for uexp_ for 64-bit signed integers in CUDA
    let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, &device);
    let mut model = BigramLanguageModel::new(vocab_size, var_builder);
    let (_, loss) = ModuleWithLoss::forward(&model, &xb, &yb)?;
    println!("{}", loss);
    let generated = model.generate(&Tensor::zeros((1, 1), DType::U32, &device)?, 100, &mut rng)?;
    println!("{}", translator.u32_array_to_chars(&generated.to_vec2::<u32>()?[0]).unwrap().iter().collect::<String>());

    let mut optimizer = candle_nn::optim::AdamW::new_lr(var_map.all_vars(), LEARNING_RATE)?;

    let batch_size = 32;
    for steps in 0..TRAINING_ITERATIONS {
        if steps % EVAL_ITERATIONS == 0 {
            let (train_loss, val_loss) = estimate_loss(&model, &device, &train_data, &validation_data, block_size, batch_size, &mut rng)?;
            println!("Iteration: {} Training Loss: {} Validation Loss: {}", steps, train_loss, val_loss);
        }
        let (logits, targets) = get_batch(&train_data, &device, batch_size, block_size, &mut rng)?;

        let (_, loss) = ModuleWithLoss::forward(&model, &logits, &targets)?;
        optimizer.backward_step(&loss)?;
    }

    let generated = model.generate(&Tensor::zeros((1, 1), DType::U32, &device)?, 100, &mut rng)?;
    println!("{}", translator.u32_array_to_chars(&generated.to_vec2::<u32>()?[0]).unwrap().iter().collect::<String>());
    Ok(())
}

fn get_batch(data: &[u32], device: &Device, batch_size: usize, block_size: usize, rng: &mut StdRng) -> candle_core::Result<(Tensor, Tensor)> {
    // builds the input values. Each set of these are associated with an index of the target
    // values, which serves as the "correct" answer to that set of inputs.
    let mut xx = vec![vec![0u32; block_size]; batch_size];
    let mut yy = vec![vec![0u32; block_size]; batch_size];
    for (_, (x, y)) in xx.iter_mut().zip(yy.iter_mut()).enumerate() {
        // generate random location within data to use as block starting location
        let start_val = rng.gen_range(0..batch_size + block_size);
        x.copy_from_slice(&data[start_val..start_val+block_size]);
        y.copy_from_slice(&data[start_val+1..start_val+block_size+1]);
    }
    Ok((Tensor::new(xx, device)?, Tensor::new(yy, device)?))
}

fn estimate_loss(model: &BigramLanguageModel, device: &Device, train_data: &[u32], val_data: &[u32], block_size: usize, batch_size: usize, rng: &mut StdRng) -> candle_core::Result<(f32, f32)> {
    let losses = (0..EVAL_ITERATIONS).map(|_| {
        let (train_logits, train_targets) = get_batch(train_data, device, batch_size, block_size, rng).unwrap();
        let (val_logits, val_targets) = get_batch(val_data, device, batch_size, block_size, rng).unwrap();
        let (_, train_loss) = ModuleWithLoss::forward(model, &train_logits, &train_targets).unwrap();
        let (_, val_loss) = ModuleWithLoss::forward(model, &val_logits, &val_targets).unwrap();
        (train_loss.to_scalar::<f32>().unwrap(), val_loss.to_scalar::<f32>().unwrap())
    }).collect::<Vec<(f32, f32)>>();
    let (train_losses, val_losses): (Vec<_>, Vec<_>) = losses.iter().cloned().unzip();
    Ok((train_losses.iter().sum::<f32>() / train_losses.len() as f32, val_losses.iter().sum::<f32>() / val_losses.len() as f32))
}

trait ModuleWithLoss {
    fn forward(&self, i: &Tensor, xs: &Tensor) -> candle_core::Result<(Tensor, Tensor)>;
}

struct BigramLanguageModel {
    // light wrapper over a tensor, meant for retrieving rows
    token_embedding_table: Embedding
}

impl BigramLanguageModel {
    pub fn new(vocab_size: usize, var_builder: VarBuilder) -> Self {
        Self {
            token_embedding_table: embedding(vocab_size, vocab_size, var_builder).unwrap()
        }
    }

    pub fn generate(&mut self, i: &Tensor, max_tokens: u32, rng: &mut StdRng) -> candle_core::Result<Tensor> {
        let mut generated = i.clone();
        for _ in 0..max_tokens {
            let logits = generated.apply(self)?;
            let logits = logits.i((.., logits.dim(1)? - 1, ..))?;
            let probabilities = candle_nn::ops::softmax(&logits, 1)?;
            let distribution = rand::distributions::WeightedIndex::new(&probabilities.to_vec2::<f32>()?[0]).unwrap();
            let i_next = distribution.sample(rng) as u32;
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
        let logits = i.apply(&self.token_embedding_table)?;
        let (b, t, c) = logits.dims3()?;
        let logits = logits.reshape((b*t, c))?;
        let targets = xs.reshape(b*t)?;
        // in this case, we expect loss to be -ln(1/65), so negative natlog of the inverse vocab size
        let loss = candle_nn::loss::cross_entropy(&logits, &targets)?;
        Ok((logits, loss))
    }
}

impl Clone for BigramLanguageModel {
    fn clone(&self) -> Self {
        Self {
            token_embedding_table: self.token_embedding_table.clone()
        }
    }
}