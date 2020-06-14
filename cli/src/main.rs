use std::path::Path;

use chinese_ner::{ChineseNER, NERTrainer};
use clap::{App, Arg, ArgMatches, SubCommand};

fn train(args: &ArgMatches) {
    let dataset_path = args.value_of("dataset").unwrap();
    if !Path::new(&dataset_path).exists() {
        panic!("Training dataset does not exist");
    }
    let mut trainer = NERTrainer::new("ner.model");
    trainer.train(dataset_path).expect("Training errored");
}

fn predict(args: &ArgMatches) {
    let text = args.value_of("text").unwrap();
    let ner = ChineseNER::new();
    let result = ner.predict(text).unwrap();
    println!("{:#?}", result);
}

fn main() {
    let mut app = App::new("chinese-ner")
        .version("0.1")
        .author("Messense Lv <messense@icloud.com>")
        .about("chinese-ner command line utility")
        .subcommand(
            SubCommand::with_name("train")
                .arg(
                    Arg::with_name("dataset")
                        .value_name("DATASET_PATH")
                        .help("Training dataset path")
                        .required(true)
                        .takes_value(true),
                )
                .help("Train a new NER model"),
        )
        .subcommand(
            SubCommand::with_name("predict")
                .arg(
                    Arg::with_name("text")
                        .value_name("TEXT")
                        .help("Text to predict")
                        .required(true)
                        .takes_value(true),
                )
                .help("Predict named entity"),
        );
    match app.clone().get_matches().subcommand() {
        ("train", Some(sub_matches)) => train(sub_matches),
        ("predict", Some(sub_matches)) => predict(sub_matches),
        _ => {
            app.print_help().unwrap();
            println!();
        }
    }
}
