extern crate crfsuite;
extern crate jieba_rs;

use std::fs::File;
use std::path::Path;
use std::io::prelude::*;
use std::io::BufReader;

use jieba_rs::Jieba;

#[derive(Debug)]
pub struct ChineseNER {
    model: crfsuite::Model,
    segmentor: jieba_rs::Jieba,
}

impl Default for ChineseNER {
    fn default() -> ChineseNER {
        ChineseNER::new()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NamedEntity<'a> {
    word: Vec<&'a str>,
    tag: Vec<&'a str>,
    entity: Vec<(usize, usize, &'static str)>,
}

impl ChineseNER {
    pub fn new() -> Self {
        let model_bytes = include_bytes!("ner.model");
        let model = crfsuite::Model::from_memory(&model_bytes[..]).expect("open model failed");
        Self {
            model,
            segmentor: Jieba::new(),
        }
    }

    pub fn from_model(model_path: &str) -> Self {
        let model = crfsuite::Model::from_file(model_path).expect("open model failed");
        Self {
            model,
            segmentor: Jieba::new(),
        }
    }

    pub fn predict<'a>(&'a self, sentence: &'a str) -> NamedEntity<'a> {
        use crfsuite::Attribute;

        let mut tagger = self.model.tagger().unwrap();
        let (split_words, tags) = split_by_words(&self.segmentor, sentence);
        let features = sent2features(&split_words);
        let attributes: Vec<crfsuite::Item> = features
            .into_iter()
            .map(|x| x.into_iter().map(|f| Attribute::new(f, 1.0)).collect::<crfsuite::Item>())
            .collect();
        let tag_result = tagger.tag(&attributes).unwrap();
        let mut is_tag = false;
        let mut start_index = 0;
        let mut entities = Vec::new();
        for (index, tag) in tag_result.iter().enumerate() {
            if !is_tag && tag.starts_with('B') {
                start_index = index;
                is_tag = true;
            } else if is_tag && tag == "O" {
                entities.push((start_index, index, get_tag_name(&tag_result[start_index])));
                is_tag = false;
            }
        }
        let words = tags.iter().map(|x| x.word).collect();
        let tags = tags.iter().map(|x| x.tag).collect();
        NamedEntity {
            word: words,
            tag: tags,
            entity: entities,
        }
    }
}

fn get_tag_name(tag: &str) -> &'static str {
    if tag.contains("PRO") {
        "product_name"
    } else if tag.contains("PER") {
        "person_name"
    } else if tag.contains("TIM") {
        "time"
    } else if tag.contains("ORG") {
        "org_name"
    } else if tag.contains("LOC") {
        "location"
    } else {
        "unknown"
    }
}

#[derive(Debug, PartialEq)]
struct SplitWord<'a> {
    word: &'a str,
    status: &'static str,
    tag: String,
    entity_type: String,
}

fn split_by_words<'a>(segmentor: &'a Jieba, sentence: &'a str) -> (Vec<SplitWord<'a>>, Vec<jieba_rs::Tag<'a>>) {
    let mut words = Vec::new();
    let mut char_indices = sentence.char_indices().map(|x| x.0).peekable();
    while let Some(pos) = char_indices.next() {
        if let Some(next_pos) = char_indices.peek() {
            let word = &sentence[pos..*next_pos];
            words.push(SplitWord {
                word: word,
                status: "",
                tag: String::new(),
                entity_type: String::new(),
            });
        } else {
            let word = &sentence[pos..];
            words.push(SplitWord {
                word: word,
                status: "",
                tag: String::new(),
                entity_type: String::new(),
            });
        }
    }
    let tags = segmentor.tag(sentence, true);
    let mut index = 0;
    for word_tag in &tags {
        let char_count = word_tag.word.chars().count();
        for i in 0..char_count {
            let status = {
                if char_count == 1 {
                    "S"
                } else if i == 0 {
                    "B"
                } else if i == char_count - 1 {
                    "E"
                } else {
                    "I"
                }
            };
            words[index].status = status;
            words[index].tag = word_tag.tag.to_string();
            index += 1;
        }
    }
    (words, tags)
}

fn sent2features(split_words: &[SplitWord]) -> Vec<Vec<String>> {
    let mut features = Vec::with_capacity(split_words.len());
    for i in 0..split_words.len() {
        features.push(word2features(split_words, i));
    }
    features
}

fn word2features(split_words: &[SplitWord], i: usize) -> Vec<String> {
    let split_word = &split_words[i];
    let word = split_word.word;
    let is_digit = word.chars().all(|c| c.is_ascii_digit());
    let mut features = vec![
        "bias".to_string(),
        format!("word={}", word),
        format!("word.isdigit={}", if is_digit { "True" } else { "False" }),
        format!("postag={}", split_word.tag),
        format!("cuttag={}", split_word.status),
    ];
    if i > 0 {
        let split_word1 = &split_words[i - 1];
        features.push(format!("-1:word={}", split_word1.word));
        features.push(format!("-1:postag={}", split_word1.tag));
        features.push(format!("-1:cuttag={}", split_word1.status));
    } else {
        features.push("BOS".to_string());
    }
    if i < split_words.len() - 1 {
        let split_word1 = &split_words[i + 1];
        features.push(format!("+1:word={}", split_word1.word));
        features.push(format!("+1:postag={}", split_word1.tag));
        features.push(format!("+1:cuttag={}", split_word1.status));
    } else {
        features.push("EOS".to_string());
    }
    features
}

pub struct NERTrainer {
    trainer: crfsuite::Trainer,
    segmentor: jieba_rs::Jieba,
    output_path: String,
}

impl NERTrainer {
    pub fn new(output_path: &str) -> Self {
        Self {
            trainer: crfsuite::Trainer::new(true),
            segmentor: Jieba::new(),
            output_path: output_path.to_string(),
        }
    }

    pub fn train<T: AsRef<Path>>(&mut self, dataset_path: T) {
        let file = File::open(dataset_path).expect("Open dataset file failed");
        let reader = BufReader::new(file);
        let lines = reader.lines().collect::<Result<Vec<String>, _>>().unwrap();
        let mut x_train = Vec::new();
        let mut y_train = Vec::new();
        let mut words: Vec<SplitWord> = Vec::new();
        for line in &lines {
            if line.is_empty() {
                let sentence: String = words.iter().map(|x| x.word).collect::<Vec<_>>().join("");
                let tags = self.segmentor.tag(&sentence, true);
                let mut index = 0;
                for word_tag in tags {
                    let char_count = word_tag.word.chars().count();
                    for i in 0..char_count {
                        let status = {
                            if char_count == 1 {
                                "S"
                            } else if i == 0 {
                                "B"
                            } else if i == char_count - 1 {
                                "E"
                            } else {
                                "I"
                            }
                        };
                        words[index].status = status;
                        words[index].tag = word_tag.tag.to_string();
                        index += 1;
                    }
                }
                x_train.push(sent2features(&words));
                y_train.push(words.iter().map(|x| x.entity_type.to_string()).collect::<Vec<_>>());
                words.clear();
            } else {
                let parts: Vec<&str> = line.split(' ').collect();
                let word = &parts[0];
                let entity_type = &parts[1];
                words.push(SplitWord {
                    word: word,
                    status: "",
                    tag: String::new(),
                    entity_type: entity_type.to_string(),
                });
            }
        }
        self.trainer.select(crfsuite::Algorithm::LBFGS, crfsuite::GraphicalModel::CRF1D).unwrap();
        for (features, yseq) in x_train.into_iter().zip(y_train) {
            let xseq: Vec<crfsuite::Item> = features
                .into_iter()
                .map(|x| x.into_iter().map(|f| crfsuite::Attribute::new(f, 1.0)).collect::<crfsuite::Item>())
                .collect();
            self.trainer.append(&xseq, &yseq, 0).unwrap();
        }
        self.trainer.train(&self.output_path, -1).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use jieba_rs::Jieba;
    use super::*;

    #[test]
    fn test_split_by_words() {
        let jieba = Jieba::new();
        let sentence = "洗衣机，国内掀起了大数据、云计算的热潮。仙鹤门地区。";
        let (ret, _) = split_by_words(&jieba, sentence);
        assert_eq!(
            ret,
            vec![
                SplitWord { word: "洗", status: "B", tag: "n".to_string(), entity_type: String::new() },
                SplitWord { word: "衣", status: "I", tag: "n".to_string(), entity_type: String::new() },
                SplitWord { word: "机", status: "E", tag: "n".to_string(), entity_type: String::new() },
                SplitWord { word: "，", status: "S", tag: "x".to_string(), entity_type: String::new() },
                SplitWord { word: "国", status: "B", tag: "s".to_string(), entity_type: String::new() },
                SplitWord { word: "内", status: "E", tag: "s".to_string(), entity_type: String::new() },
                SplitWord { word: "掀", status: "B", tag: "v".to_string(), entity_type: String::new() },
                SplitWord { word: "起", status: "E", tag: "v".to_string(), entity_type: String::new() },
                SplitWord { word: "了", status: "S", tag: "ul".to_string(), entity_type: String::new() },
                SplitWord { word: "大", status: "S", tag: "a".to_string(), entity_type: String::new() },
                SplitWord { word: "数", status: "B", tag: "n".to_string(), entity_type: String::new() },
                SplitWord { word: "据", status: "E", tag: "n".to_string(), entity_type: String::new() },
                SplitWord { word: "、", status: "S", tag: "x".to_string(), entity_type: String::new() },
                SplitWord { word: "云", status: "S", tag: "ns".to_string(), entity_type: String::new() },
                SplitWord { word: "计", status: "B", tag: "v".to_string(), entity_type: String::new() },
                SplitWord { word: "算", status: "E", tag: "v".to_string(), entity_type: String::new() },
                SplitWord { word: "的", status: "S", tag: "uj".to_string(), entity_type: String::new() },
                SplitWord { word: "热", status: "B", tag: "n".to_string(), entity_type: String::new() },
                SplitWord { word: "潮", status: "E", tag: "n".to_string(), entity_type: String::new() },
                SplitWord { word: "。", status: "S", tag: "x".to_string(), entity_type: String::new() },
                SplitWord { word: "仙", status: "B", tag: "n".to_string(), entity_type: String::new() },
                SplitWord { word: "鹤", status: "E", tag: "n".to_string(), entity_type: String::new() },
                SplitWord { word: "门", status: "S", tag: "n".to_string(), entity_type: String::new() },
                SplitWord { word: "地", status: "B", tag: "n".to_string(), entity_type: String::new() },
                SplitWord { word: "区", status: "E", tag: "n".to_string(), entity_type: String::new() },
                SplitWord { word: "。", status: "S", tag: "x".to_string(), entity_type: String::new() },
            ]
        );
    }

    #[test]
    fn test_ner_predict() {
        let ner = ChineseNER::new();
        let sentence = "今天纽约的天气真好啊，京华大酒店的李白经理吃了一只北京烤鸭。";
        let result = ner.predict(sentence);
        assert_eq!(result.word, vec!["今天", "纽约", "的", "天气", "真好", "啊", "，", "京华", "大酒店", "的", "李白", "经理", "吃", "了", "一只", "北京烤鸭", "。"]);
        assert_eq!(result.tag, vec!["t", "ns", "uj", "n", "d", "zg", "x", "nz", "n", "uj", "nr", "n", "v", "ul", "m", "n", "x"]);
        assert_eq!(result.entity, vec![(2, 4, "location"), (11, 16, "org_name"), (17, 19, "person_name"), (25, 27, "location")]);
    }
}
