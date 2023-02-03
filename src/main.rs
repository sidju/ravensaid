use anyhow::Result;
use tch::{Device, Kind, Tensor, Reduction};
use tch::nn::{Module, linear, seq, Adam, OptimizerConfig, VarStore};

const LEARNING_RATE_1: f64 = 0.000005;
const LEARNING_RATE_2: f64 = 0.000001;
const INPUT_BYTES: i64 = 32;
// A float for every permutation of every byte
const INPUT_SIZE: i64 = INPUT_BYTES * 256;
// One float per input character seems reasonable
const HIDDEN_SIZE: i64 = 64;
const EPOCHS: i64 = 100;

fn main() -> Result<()> {
  // Create device connection and a variable store on it
  let device = Device::cuda_if_available();
  let mut vs = VarStore::new(device);
  // Prepare computations by defining the neural network
  let layers = seq()
    // First layer takes the input and compresses it down to a manageable size
    .add(linear(vs.root() / "l1", INPUT_SIZE, HIDDEN_SIZE, Default::default()))
    // Second layer takes the compressed data and tries to process it
//    .add(linear(vs.root() / "l2", HIDDEN_SIZE, HIDDEN_SIZE, Default::default()))
    // Third layer tries to boil it all down into one value, the one we want
    .add(linear(vs.root() / "l3", HIDDEN_SIZE, 1, Default::default()))
  ;
  
  // Either load trained network or train and output a network
  let mut args = std::env::args();
  args.next(); // Throw away our executable's name
  match args.next().as_ref().map(|s| &s[..]) {
    Some("train") => train_nn(&layers, &mut vs, device, args.next().unwrap_or(String::new()))?,
    Some("run") => {
      if let Some(s) = args.next() {
        vs.load(s)?;
        run_nn_on_input(&layers)?;
      }
      else {
        eprintln!("You must give path to the model to run.");
      }
    }
    None | Some(_) => {
      println!("You need to give a command. Either 'run' or 'train'.");
    },
  }

  Ok(())
}

fn train_nn<M: Module>(layers: &M, vs: &mut VarStore, device: Device, prefix: String) -> Result<()> {
  // Read in the training data
  // Just read it into memory
  let mut data = Vec::<(&str, bool)>::new();
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
      data.push((s, false));
    }
    else { break; }
    if let Some(s) = ri.next() {
      data.push((s, true));
    }
    else { break; }
    if let Some(s) = di.next() {
      data.push((s, false));
    }
    else { break; }
  }
  let training_data = data.split_off(data.len() / 5);
  let validation_data = data;
  println!("Added {} sentences to train on and {} to validate with.",
    training_data.len(),
    validation_data.len(),
  );

  // Training
  let mut optimiser = Adam::default().build(&vs, LEARNING_RATE_1)?;
  for epoch in 0..EPOCHS {
    if epoch == EPOCHS / 2 { optimiser.set_lr(LEARNING_RATE_2); }
    let mut sum_loss = 0.;
    let mut cnt_loss = 0.;

    // Calculate selection to train on
    let len = training_data.len() / 10;
    let starting_point = (epoch as usize % 10) * len;

    for (k,v) in &training_data[starting_point .. starting_point + len] {
      let predicted = layers.forward(&sentence_to_tensor(k).to_device(device));
      let loss = predicted
        .binary_cross_entropy_with_logits::<Tensor>(
          &bool_to_tensor(*v).to_device(device),
          None,
          None,
          Reduction::Mean,
        )
      ;
      // Clipping argument 0.5 should be fine
      optimiser.backward_step_clip(&loss, 0.5 as f64);
      sum_loss += f64::from(loss);
      cnt_loss += 1.;
    }
    println!("Prefix: {}, epoch: {}, loss: {}", prefix, epoch, sum_loss / cnt_loss);
    test(layers, &validation_data);
    vs.save(format!("./{prefix}epoch_{epoch}.nn"))?;
  }
  // Write out the trained values to a file
  Ok(())
}

fn run_nn_on_input<M: Module>(layers: &M) -> Result<()> {
  println!("Enter messages to estimate likelyhood that Raven said:");
  let stdin = std::io::stdin();
  loop {
    let mut input_line = String::new();
    let read = stdin.read_line(&mut input_line)?;
    if read == 0 { continue; }
    input_line.pop(); // Get rid of newline
    if &input_line == "exit" { break; }

    let tmp = layers.forward(&sentence_to_tensor(&input_line));
    let result = f64::from(tmp.sigmoid()) * 100.;
    println!("Likelyhood that raven said ^: {:.2}%", result);
  }
  Ok(())
}

fn sentence_to_tensor(s: &str) -> Tensor {
  // Write up to 32 bytes to a Tensor
  let mut tmp: Vec<f64> = Vec::new();
  tmp.resize_with(INPUT_SIZE as usize, || 0.0);
  let bytes = s.bytes();
  for (i, b) in bytes.enumerate() {
    if i >= 32 { break; }
    tmp[i * 256 + b as usize] = 1.0;
  }
  Tensor::of_slice(&tmp[..]).to_kind(Kind::Float)
}
fn bool_to_tensor(b: bool) -> Tensor {
  let tmp: [f64; 1] = [if b { 1. } else { 0. }];
  Tensor::of_slice(&tmp[..]).to_kind(Kind::Float)
}
fn test(layers: &dyn Module, data: &[(&str, bool)]) {
  let mut successes = 0;
  let mut tests = 0;

  for (k,v) in data {
    let tmp = layers.forward(&sentence_to_tensor(k));
    let out = bool::from(tmp.sigmoid().round());
    tests += 1;
    if out == *v {
      successes += 1;
    }
  }
  println!(
    "Tested on {} sentences, {} passed, {:.2}% success",
    tests,
    successes,
    100. * successes as f64 / tests as f64,
  );
}
