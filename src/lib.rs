use std::ffi::CStr;
use std::ptr::null_mut; // "constructor" for null, the error return for init

use tch::{
  Device,
  Kind,
  Tensor,
  nn::{
    linear,
    seq,
    Sequential,
    VarStore,
    Module,
  },
};

const INPUT_BYTES: i64 = 32;
const INPUT_SIZE: i64 = INPUT_BYTES * 256;
const HIDDEN_SIZE: i64 = 64;


pub struct RavensaidState {
  // This variable is just here to prevent freeing, as this is used by layers
  // behind the scenes
  #[allow(dead_code)]
  vs: VarStore,
  layers: Sequential,
}

/// Creates state for Ravensaid on heap and returns pointer
/// Returns null pointer on error
#[no_mangle]
pub extern "C" fn ravensaid_init(path: *const libc::c_char) -> *mut RavensaidState {
  // Verify and convert path to rust usable shape
  if path.is_null() {
    return null_mut();
  }
  let path = unsafe { match CStr::from_ptr(path).to_str() {
    Ok(p) => p,
    Err(_) => return null_mut(),
  }};

  // Allocate and read in neural network from given path
  let device = Device::cuda_if_available();
  let mut vs = VarStore::new(device);
  let layers = seq()
    .add(linear(vs.root() / "l1", INPUT_SIZE, HIDDEN_SIZE, Default::default()))
    .add(linear(vs.root() / "l3", HIDDEN_SIZE, 1, Default::default()))
  ;
  if let Err(_) = vs.load(path) {
    return null_mut();
  }

  Box::into_raw(Box::new(RavensaidState{
    vs,
    layers,
  }))
}

#[no_mangle]
pub unsafe extern "C" fn ravensaid_free(state: *mut RavensaidState) {
  if state.is_null() {
    panic!("You can't drop Null");
  }
  // BEWARE!!! Renaming this variable to _ will skip allocation, and thus the
  // function won't free the memory.
  let _tmp = Box::from_raw(state); // Memory is freed on return
}

// Returns fixed point percentage with two decimals estimating how likely it is
// that Ravenholdt sent the given message
// Returns -1 on bad message formatting
// Returns -2 if the NN went off the rails in a crashing way
#[no_mangle]
pub extern "C" fn ravensaid(
  state: *mut RavensaidState,
  message: *const libc::c_char,
) -> i32 {
  // Verify and convert message to rust usable shape
  if message.is_null() {
    return -1;
  }
  let message = match unsafe { CStr::from_ptr(message).to_str() } {
    Ok(m) => m,
    Err(_) => return -1,
  };

  // Convert message to a valid input tensor
  let input_tensor = {
    let mut tmp: Vec<f64> = Vec::new();
    tmp.resize_with(INPUT_SIZE as usize, || 0.0);
    let bytes = message.bytes();
    for (i, b) in bytes.enumerate() {
      if i >= 32 { break; }
      tmp[i * 256 + b as usize] = 1.0;
    }
    Tensor::of_slice(&tmp[..]).to_kind(Kind::Float)
  };

  // Send the input through the NN
  let output_tensor = ( unsafe { &*state }).layers.forward(&input_tensor);

  // Convert to fixed point int with two decimal places
  let output = f64::from(output_tensor.sigmoid());
  if output > 2.0 { return -2; }
  if output < 0.0 { return -3; }
  let fixed_point_percent = (output * 100. * 100.) as i32;
  fixed_point_percent
}
