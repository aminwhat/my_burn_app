#[derive(Module, Debug)]
struct SimpleModel {
    layer: Linear,
}

impl SimpleModel {
    fn new() -> Self {
        Self {
            // The "magic math box" takes an input size of 2 and outputs size 1.
            layer: Linear::new(&LinearConfig::new(2, 1)),
        }
    }

    fn forward(&self, input: Tensor<2>) -> Tensor<2> {
        // Pass the input through the "magic math box"
        self.layer.forward(input)
    }
}
