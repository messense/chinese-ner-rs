# chinese-ner-rs

[![Crates.io](https://img.shields.io/crates/v/chinese-ner.svg)](https://crates.io/crates/chinese-ner)
[![docs.rs](https://docs.rs/chinese-ner/badge.svg)](https://docs.rs/chinese-ner/)

A CRF based Chinese Named-entity Recognition Library written in Rust

## Installation

Add it to your ``Cargo.toml``:

```toml
[dependencies]
chinese-ner = "0.1"
```

Add ``extern crate chinese_ner`` to your crate root and your're good to go!

## Example

```rust
extern crate chinese_ner;

use chinese_ner::ChineseNER;

fn main() {
    let ner = ChineseNER::new();
    let result = ner.predict("今天上海天气很好").unwrap();
    println!("{:?}", result);
}
```

## License

This work is released under the MIT license. A copy of the license is provided in the [LICENSE](./LICENSE) file.
