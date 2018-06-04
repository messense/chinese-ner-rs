extern crate chinese_ner;

use std::env;
use chinese_ner::NERTrainer;

fn main() {
    let dataset_path = env::args().skip(1).next().expect("No training dataset provided");
    let mut trainer = NERTrainer::new("ner.model");
    trainer.train(dataset_path);
}
