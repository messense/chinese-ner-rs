extern crate crfsuite;
extern crate jieba_rs;

use jieba_rs::Jieba;

#[derive(Debug)]
pub struct ChineseNER {
    model: crfsuite::Model,
    segmentor: jieba_rs::Jieba,
}

impl ChineseNER {
    pub fn new(model_path: &str) -> Self {
        let model = crfsuite::Model::from_file(model_path).expect("open model failed");
        Self {
            model,
            segmentor: Jieba::new(),
        }
    }

    pub fn predict(&self, sentence: &str) -> Vec<String> {
        use crfsuite::Attribute;

        let mut tagger = self.model.tagger().unwrap();
        let split_words = split_by_words(&self.segmentor, sentence);
        let features = sent2features(&split_words);
        let attributes: Vec<crfsuite::Item> = features
            .into_iter()
            .map(|x| x.into_iter().map(|f| Attribute::new(f, 1.0)).collect::<crfsuite::Item>())
            .collect();
        let tag_result = tagger.tag(&attributes).unwrap();
        tag_result
    }
}

#[derive(Debug, PartialEq)]
struct SplitWord<'a> {
    word: &'a str,
    status: &'static str,
    tag: &'a str,
}

fn split_by_words<'a>(segmentor: &'a Jieba, sentence: &'a str) -> Vec<SplitWord<'a>> {
    let mut words = Vec::new();
    let mut char_indices = sentence.char_indices().map(|x| x.0).peekable();
    while let Some(pos) = char_indices.next() {
        if let Some(next_pos) = char_indices.peek() {
            let word = &sentence[pos..*next_pos];
            words.push(SplitWord {
                word: word,
                status: "",
                tag: "",
            });
        } else {
            let word = &sentence[pos..];
            words.push(SplitWord {
                word: word,
                status: "",
                tag: "",
            });
        }
    }
    let tags = segmentor.tag(sentence, true);
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
            words[index].tag = word_tag.tag;
            index += 1;
        }
    }
    words
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

#[cfg(test)]
mod tests {
    use jieba_rs::Jieba;
    use super::*;

    #[test]
    fn test_split_by_words() {
        let jieba = Jieba::new();
        let sentence = "洗衣机，国内掀起了大数据、云计算的热潮。仙鹤门地区。";
        let ret = split_by_words(&jieba, sentence);
        assert_eq!(
            ret,
            vec![
                SplitWord { word: "洗", status: "B", tag: "n" },
                SplitWord { word: "衣", status: "I", tag: "n" },
                SplitWord { word: "机", status: "E", tag: "n" },
                SplitWord { word: "，", status: "S", tag: "x" },
                SplitWord { word: "国", status: "B", tag: "s" },
                SplitWord { word: "内", status: "E", tag: "s" },
                SplitWord { word: "掀", status: "B", tag: "v" },
                SplitWord { word: "起", status: "E", tag: "v" },
                SplitWord { word: "了", status: "S", tag: "ul" },
                SplitWord { word: "大", status: "S", tag: "a" },
                SplitWord { word: "数", status: "B", tag: "n" },
                SplitWord { word: "据", status: "E", tag: "n" },
                SplitWord { word: "、", status: "S", tag: "x" },
                SplitWord { word: "云", status: "S", tag: "ns" },
                SplitWord { word: "计", status: "B", tag: "v" },
                SplitWord { word: "算", status: "E", tag: "v" },
                SplitWord { word: "的", status: "S", tag: "uj" },
                SplitWord { word: "热", status: "B", tag: "n" },
                SplitWord { word: "潮", status: "E", tag: "n" },
                SplitWord { word: "。", status: "S", tag: "x" },
                SplitWord { word: "仙", status: "B", tag: "n" },
                SplitWord { word: "鹤", status: "E", tag: "n" },
                SplitWord { word: "门", status: "S", tag: "n" },
                SplitWord { word: "地", status: "B", tag: "n" },
                SplitWord { word: "区", status: "E", tag: "n" },
                SplitWord { word: "。", status: "S", tag: "x" },
            ]
        );
    }

    #[test]
    fn test_ner_predict() {
        let ner = ChineseNER::new("ner.model");
        let sentence = "今天纽约的天气真好啊，京华大酒店的李白经理吃了一只北京烤鸭。";
        let tags = ner.predict(sentence);
        assert_eq!(tags, vec!["O", "O", "B-LOC", "I-LOC", "O", "O", "O", "O", "O", "O", "O", "B-ORG", "I-ORG", "I-ORG", "I-ORG", "I-ORG", "O", "B-PER", "I-PER", "O", "O", "O", "O", "O", "O", "B-LOC", "I-LOC", "O", "O", "O"]);
    }
}
