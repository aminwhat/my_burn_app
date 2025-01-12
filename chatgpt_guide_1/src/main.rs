use burn::loss::Loss;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{Optimizer, Sgd, SgdConfig};
use burn::tensor::Tensor;
use burn::train::{LearnerBuilder, TrainOutput};

mod model;

fn main() {
    // Step 1: Create the model
    let model = SimpleModel::new();

    // Step 2: Pretend data
    let inputs = Tensor::from_data([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    let targets = Tensor::from_data([[3.0], [7.0], [11.0]]); // What we want the output to be.

    // Step 3: Create optimizer (helps adjust model)
    let mut optimizer = Sgd::new(&SgdConfig::new(0.01)); // Learning rate 0.01

    // Step 4: Train for 100 steps
    for step in 0..100 {
        // Forward pass (make a guess)
        let outputs = model.forward(inputs.clone());

        // Calculate the error (loss)
        let loss = outputs.mse_loss(targets.clone());

        // Backward pass (fix the guess)
        optimizer.update(&model, &loss);

        // Print the loss every 10 steps
        if step % 10 == 0 {
            println!("Step {step}: Loss = {:?}", loss);
        }
    }

    // Done! Test it again:
    let test_output = model.forward(inputs);
    println!("Final Output: {:?}", test_output);
}
