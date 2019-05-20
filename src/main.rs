use std::env;
use std::path::Path;

use chinese_ner::NERTrainer;

fn main() {
    let dataset_path = env::args().skip(1).next().expect("No training dataset provided");
    if !Path::new(&dataset_path).exists() {
        panic!("Training dataset does not exist");
    }
    let mut trainer = NERTrainer::new("ner.model");
    trainer.train(dataset_path).expect("Training errored");
}
