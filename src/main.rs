// use std::io;
// use std::io::prelude::*;
// use std::fs::File;
// use nalgebra as na;

fn main() {
	// The set of inputs we want to mimic
	let inputs: Vec<f64> = vec![0.0, 0.0, 10.0, 0.0, 1.0, 0.0, 1.0];
	let target = 30.0;
	let learning_rate = 0.1;
	
	// Create a neural net that learns from our inputs.
	// The 2nd argument is the target weight that node must have in order to fire.
	// The 3rd argument is the amount weights increase by every time learn() is called.
	let mut net = Net::new(inputs, target, learning_rate);
	
	// Run 6 trials.
	println!("{:?}", net.run(9).0);
}

struct Net {
	index: usize,
	nodes: Vec<Node>,
	target: f64,
	learning_rate:f64,
}

impl Net {
	pub fn new(inputs: Vec<f64>, target: f64, learning_rate: f64) -> Net {
		let mut net = Net {
			index: 0,
			nodes: vec!(),
			target: target,
			learning_rate: learning_rate,
		};
		net.initialize(inputs);
		net
	}
	// Create nodes from inputs.
	fn initialize(&mut self, inputs: Vec<f64>) {
		for input in inputs {
			self.add_node(input);
		}
	}
	
	pub fn run(&mut self, trials: usize) -> (Vec<f64>, Option<usize>) {
		for trial in 0..trials {
			// Evaluate the sum of the inputs multiplied by their weights.
			let result = self.evaluate();
			// See how far off that sum is from our target.
			let error = self.evaluate_err(result);
			println!("Trial {}: Result: {}. Error: {}.", trial, result, error);
			// If we have reached our target,
			if error <= 0.0 {
				return (self.weights(), Some(trial));
			}
			// If not, adjust our weights.
			self.learn();
		}
		(self.weights(), None)
	}
	// For each node, multiply the node input by its weight and add that to the result.
	fn evaluate(&mut self) -> f64 {
		let mut result = 0.0;
		while let Some(node) = self.next() {
			result += node.input() * node.weight();
		}
		// result = 1.0 - (1.0 / result); // Sigmoid
		result
	}
	
	fn evaluate_err(&self, real: f64) -> f64 {
		self.target() - real
	}
	// For each node, if the input of that node is greater than zero, increase its weight by the learning_rate.
	fn learn(&mut self) {
		let learning_rate = self.learning_rate();
		while let Some(mut node) = self.next() {
			if node.input() > 0.0 {
				node.weight += learning_rate;
			}
		}
	}
	// I made an iterator because it makes looping look nicer.
	fn next(&mut self) -> Option<&mut Node> {
		if self.len() > self.index() {
			let i = self.index();
			self.index += 1;
			return Some(&mut self.nodes[i]);
		}
		self.index = 0;
		None
	}
	
	fn add_node(&mut self, input: f64) {
		self.nodes.push(Node::new(input));
	}
	
	fn len(&self) -> usize {
		self.nodes.len()
	}
	
	fn index(&self) -> usize {
		self.index.clone()
	}
	
	fn target(&self) -> f64 {
		self.target.clone()
	}
	
	fn learning_rate(&self) -> f64 {
		self.learning_rate.clone()
	}
	// Make a vec from all the weights of all the nodes.
	// I could have put the weights in different lists, but I figured that keeping data that gets used
	// together should stay together, so that's why I made this.
	fn weights(&mut self) -> Vec<f64> {
		let mut result_vec: Vec<f64> = vec!();
		while let Some(node) = self.next() {
			result_vec.push(node.weight());
		}
		result_vec
	}
}

struct Node {
	input: f64,
	weight: f64,
}

impl Node {
	pub fn new(input: f64) -> Node {
		Node {
			input: input,
			weight: 0.0,
		}
	}
	
	pub fn input(&self) -> f64 {
		self.input.clone()
	}
	
	pub fn weight(&self) -> f64 {
		self.weight.clone()
	}
}
