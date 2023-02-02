use anyhow::Result;
use tch::{nn, Device, Kind, Tensor, Reduction};
use tch::nn::{Module, linear, seq, Sequential, Adam, OptimizerConfig};

const LEARNING_RATE: f64 = 0.01;
const INPUT_BYTES: i64 = 32;
// A float for every permutation of every byte
const INPUT_SIZE: i64 = INPUT_BYTES * 256;
const HIDDEN_SIZE: i64 = 128 * 256;
const EPOCHS: i64 = 100;

fn main() -> Result<()> {
  // Create device connection and a variable store on it
  let device = Device::cuda_if_available();
  let vs = nn::VarStore::new(device);
  // Prepare computations by defining the neural network
  let layers = seq()
    // First layer takes up to 32 bytes of utf-8, giving out 128 values.
    .add(linear(vs.root() / "l1", INPUT_SIZE, HIDDEN_SIZE, Default::default()))
    // Second layer keeps width, adding some more processing potential.
//    .add(linear(vs.root() / "l2", HIDDEN_SIZE, HIDDEN_SIZE, Default::default()))
    // Third layer tries to boil it all down into one value, the one we want
    .add(linear(vs.root() / "l3", HIDDEN_SIZE, 1, Default::default()))
  ;

  // Read in the training data
  // Just read it into memory
  let mut data = std::collections::HashMap::<&str, bool>::new();
  use std::fs::read_to_string;
  let berk = read_to_string("data/berk.txt")?;
  let raven = read_to_string("data/ravenholdt.txt")?;
  let dreamer = read_to_string("data/dreamer.txt")?;
  let sidju = read_to_string("data/sidju.txt")?;

  // Chain sidju with berk, since they are both shorter than raven
  let mut bi = berk.split("\n\n").chain(sidju.split("\n\n"));
  let mut ri = raven.split("\n\n");
  let mut di = dreamer.split("\n\n");

  // For each raven quote add one other quote after, to balance the learning
  loop {
    // Take one sentence from each until one is empty
    if let Some(s) = bi.next() {
      data.insert(s, false);
    }
    else { break; }
    if let Some(s) = ri.next() {
      data.insert(s, true);
    }
    else { break; }
    if let Some(s) = di.next() {
      data.insert(s, false);
    }
    else { break; }
  }
  println!("Added {} sentences to train on.", data.len());

  // Training
  let mut optimiser = Adam::default().build(&vs, LEARNING_RATE)?;
  for epoch in 0..EPOCHS {
    let mut sum_loss = 0.;
    let mut cnt_loss = 0.;

    for k in data.keys() {
      let correct = data.get(k).unwrap();
      println!("\n\nIt currently crashes right after here\n\n");
      let predicted = layers.forward(&sentence_to_tensor(k).to_device(device));
      let loss = predicted.view([1])
        .binary_cross_entropy::<Tensor>(
          &bool_to_tensor(*correct).to_device(device).view([1]),
          None,
          Reduction::Mean,
        )
      ;
      optimiser.backward_step_clip(&loss, 0.5 as f64);
      sum_loss += f64::from(loss);
      cnt_loss += 1.;
    }
    println!("Epoch: {}, loss: {}", epoch, sum_loss / cnt_loss);
    test(&layers);
  }

  Ok(())
}

fn sentence_to_tensor(s: &str) -> Tensor {
  // Write up to 32 bytes to a Tensor
//  Tensor::of_slice(&s[ .. s.len().min(INPUT_SIZE as usize)].as_bytes())
  let mut tmp: Vec<f64> = Vec::new();
  tmp.resize_with(INPUT_SIZE as usize, || 0.0);
  for (i, b) in (&s[..s.len().min(INPUT_BYTES as usize)]).bytes().enumerate() {
    tmp[i * 256 + b as usize] = 1.0;
  }
  Tensor::of_slice(&tmp[..]).to_kind(Kind::Float)
}
fn bool_to_tensor(b: bool) -> Tensor {
  let tmp: [f64; 1] = [if b { 1. } else { 0. }];
  Tensor::of_slice(&tmp[..]).to_kind(Kind::Int64)
}

fn test(layers: &Sequential) {
  let tests = std::collections::HashMap::from([
    ("That plot twist.", true),
    ("all according to keikaku", false),
  ]);

  for key in tests.keys() {
    let tmp = layers.forward(&sentence_to_tensor(key));
    let out = tmp.double_value(&[1]);
    println!(
      "Tested on \"{}\", correct answer is {}, NN gave {}",
      key,
      tests.get(key).unwrap(),
      out,
    );
  }
}
