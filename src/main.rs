// Stuff I'd like to do:
// Most common layer types, easy addition of new types
// Fluent, iterator-like API
// Snapshots
// Serialization
// Larger-than-resident-memory nets
// SLI
// Cross-machine
// Heterogenous networks

#![feature(convert)]
extern crate rand;
use rand::Rng;
use std::iter::{repeat};
use std::cell::RefCell;

fn zero_vec(size: usize) -> Vec<f32> {
    repeat(0.0).
        take(size).
        collect()
}

fn random_vec(size: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();

    (0..size).
        map(|_| rng.gen::<f32>()).
        collect()
}

struct Layer {
    size: usize,
    neurons: Vec<f32>,
    deltas: Vec<f32>,
    errors: Vec<f32>,
    biases: Vec<f32>,
    weights: Vec<f32>,
    changes: Vec<f32>
}

impl Layer {
    fn new(size: usize, input_size: usize) -> Layer {
        Layer {
            size: size,
            neurons: zero_vec(size),
            deltas: zero_vec(size),
            errors: zero_vec(size),
            biases: random_vec(size),
            weights: random_vec(size * input_size),
            changes: zero_vec(size * input_size)
        }
    }
}

struct BasicSolver {
    target_output: Vec<f32>,
    learning_rate: f32,
    momentum: f32
}

impl BasicSolver {
    fn new(learning_rate: f32, momentum: f32) -> BasicSolver {
        BasicSolver {
            target_output: Vec::new(),
            learning_rate: learning_rate,
            momentum: momentum
        }
    }

    fn feed_forward(&self, input_layer: &Layer, output_layer: &mut Layer) -> f32 {
        let input_size = input_layer.size;
        let output_size = output_layer.size;
        let mut result = 0.0;

        for i in 0 .. output_size {
            let bias = output_layer.biases[i];
            let sum = input_layer.neurons.iter().enumerate().fold(bias, |sum, (j, input_neuron)| {
                let weight_index = (input_size * i) + j;
                sum + output_layer.weights[weight_index] * input_neuron
            });

            result = 1.0 / (1.0 + (-sum).exp());
            output_layer.neurons[i] = result;
        }

        result
    }

    fn calculate_deltas(&self, input_layer: Option<&RefCell<Layer>>, output_layer: &mut Layer) {
        let output_size = output_layer.size;

        for i in 0 .. output_size {
            let neuron = output_layer.neurons[i];
            let error = match input_layer {
                Some(ref layer_cell) => {
                    let layer = layer_cell.borrow();
                    layer.deltas.iter().enumerate().fold(0.0, |sum, (j, delta)| {
                        let weight_index = (output_size * j) + i;
                        sum + (delta * layer.weights[weight_index])
                    })
                },
                None => self.target_output[i] - neuron
            };

            output_layer.errors[i] = error;
            output_layer.deltas[i] = error * neuron * (1.0 - neuron);
        }
    }

    fn adjust_weights(&self, input_layer: &Layer, output_layer: &mut Layer) {
        let input_size = input_layer.size;
        let output_size = output_layer.size;
        let learning_rate = self.learning_rate;
        let momentum = self.momentum;

        for i in 0 .. output_size {
            let delta = output_layer.deltas[i];

            for (j, neuron) in input_layer.neurons.iter().enumerate() {
                let change_index = (input_size * i) + j;
                let mut change = output_layer.changes[change_index];

                change = (learning_rate * delta * neuron) + (momentum * change);

                output_layer.changes[change_index] = change;
                output_layer.weights[change_index] += change;
            }

            output_layer.biases[i] += learning_rate * delta;
        }
    }
}

struct Network {
    solver: BasicSolver,
    layers: Vec<RefCell<Layer>>
}

impl Network {
    fn new(sizes: Vec<usize>) -> Network {
        Network {
            solver: BasicSolver::new(0.3, 0.1),
            layers: sizes.iter().
                scan(0, |prev_size, size| {
                    let result = Layer::new(*size, *prev_size);
                    *prev_size = *size;
                    Some(result)
                }).
                map(|layer| RefCell::new(layer)).
                collect()
        }
    }

    fn run(&mut self, input: &Vec<f32>) -> f32 {
        //TODO: Compare lengths

        let mut result = 0.0;
        let layers_len = self.layers.len();
        {
            let input_layer = self.layers.first().unwrap();
            input_layer.borrow_mut().neurons = input.clone();
        }

        for i in 1 .. layers_len {
            let input_layer = self.layers[i - 1].borrow();
            let mut output_layer = self.layers[i].borrow_mut();

            result = self.solver.feed_forward(&*input_layer, &mut *output_layer);
        }

        result
    }

    fn train(&mut self, data: &Vec<(Vec<f32>, Vec<f32>)>) -> f32 {
        (0 .. 20000).
            map(|_| {
                data.iter().fold(0.0, |sum, training_set| {
                    let (ref input, ref target) = *training_set;
                    sum + self.train_pattern(input, target)
                }) / data.len() as f32
            }).
            take_while(|error| *error > 0.00075).
            fold(1.0, |_, error| error)
    }

    fn train_pattern(&mut self, input: &Vec<f32>, target: &Vec<f32>) -> f32 {
        self.run(input);
        self.calculate_deltas(target);
        self.adjust_weights();

        let output_layer = &self.layers.last().unwrap().borrow();

        self.mean_squared_error(&output_layer.errors)
    }

    fn calculate_deltas(&mut self, target: &Vec<f32>) {
        self.solver.target_output = target.clone();

        let layers_len = self.layers.len();
        for i in (0 .. layers_len).rev() {
            let input_layer = self.layers.get(i + 1).map(|layer| layer);
            let mut output_layer = self.layers[i].borrow_mut();

            self.solver.calculate_deltas(input_layer, &mut *output_layer);
        }
    }

    fn adjust_weights(&mut self) {
        let layers_len = self.layers.len();
        for i in 1 .. layers_len {
            let input_layer = self.layers[i - 1].borrow();
            let mut output_layer = self.layers[i].borrow_mut();

            self.solver.adjust_weights(&*input_layer, &mut *output_layer);
        }
    }

    fn mean_squared_error(&self, errors: &Vec<f32>) -> f32 {
        errors.iter().fold(0.0, |sum, error| sum + error.powf(2.0)) / errors.len() as f32
    }
}

fn run_xor() {
    // A network which computes exclusive-or
    let mut xor_net = Network::new(vec![2, 3, 1]);

    let data = vec![
        (vec![0.0, 0.0], vec![0.0]), // 0 ^ 0 == 0
        (vec![0.0, 1.0], vec![1.0]), // 0 ^ 1 == 1
        (vec![1.0, 1.0], vec![0.0]), // 1 ^ 1 == 0
        (vec![1.0, 0.0], vec![1.0])  // 1 ^ 0 == 1
    ];

    xor_net.train(&data);

    println!("{:?}", xor_net.run(&vec![0.0, 0.0]).round());
    println!("{:?}", xor_net.run(&vec![0.0, 1.0]).round());
    println!("{:?}", xor_net.run(&vec![1.0, 1.0]).round());
    println!("{:?}", xor_net.run(&vec![1.0, 0.0]).round());
}

fn run_dec_to_bin() {
    fn make_num_vec(num: usize) -> Vec<f32> {
        let mut result = zero_vec(10);
        result[num] = 1.0;

        result
    }

    fn make_bin_vec(num: usize) -> Vec<f32> {
        (0 .. 4).
            rev(). // Gotta walk right-to-left due to little endian-ness
            map(|i| ((num >> i) & 1) as f32).
            collect()
    }

    // A network which maps decimals into binary
    let mut dec_to_bin_net = Network::new(vec![10, 4]);

    let data = (0 .. 10).
        map(|num| (make_num_vec(num), make_bin_vec(num))).
        collect();

    dec_to_bin_net.train(&data);

    for i in 0 .. 10 {
        dec_to_bin_net.run(&make_num_vec(i));
        let rounded_output = dec_to_bin_net.layers.last().unwrap().borrow().neurons.iter().
            map(|neuron| neuron.round() as i32).
            skip_while(|neuron| *neuron == 0).
            fold(String::new(), |accum, neuron| accum + neuron.to_string().as_str());

        println!("Binary: {:b} Network: {:?}", i, rounded_output);
    }
}

fn main() {
    run_xor();
    run_dec_to_bin();
}