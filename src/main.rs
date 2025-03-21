extern crate core;

mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::collections::HashMap;
use std::io;
use std::io::Write;
use std::path::PathBuf;
use tokenizers::Tokenizer;
use crate::kvcache::KVCache;
use crate::model::Llama;

pub struct ChatManager {
    pub messages: Vec<String>,
    pub kv_cache: KVCache<f32>,
    pub history: HashMap<u32, Vec<u32>>,
    pub llama: Llama<f32>,
    pub tokenizer: Tokenizer,
}

impl ChatManager {
    fn new(llama: Llama<f32>, tk: Tokenizer) -> Self {
        ChatManager {
            messages: vec![],
            kv_cache: llama.new_cache(),
            history: HashMap::new(),
            llama,
            tokenizer: tk,
        }
    }

    fn run(&mut self) {
        loop {
            self.messages
                .push("user: hello. AI is chatting with user\n".to_string());

            self.messages.push("AI: nice to meet you\n".to_string());

            self.messages
                .push("user: please chat wite me\n".to_string());

            self.messages.push("AI: OK\n".to_string());


            print!("user: ");
            io::stdout().flush().unwrap();
            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            let input = input.trim();

            let prompt = self.messages.join("") + "AI: ";
            let binding = self.tokenizer.encode(prompt, false).unwrap();
            let input_ids = binding.get_ids();

            let output_ids = self.llama.answer(input_ids, 100, 0.8, 30, 1., &mut self.kv_cache);
            let resp = self.tokenizer.decode(&output_ids, true).unwrap();

            self.messages
                .push(format!("AI: {}\n", resp));
            println!("AI: {}", resp);
        }
    }
}



fn main() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let input = "Once upon a time";
    let binding = tokenizer.encode(input, false).unwrap();
    let input_ids = binding.get_ids();
/*    print!("\n{}", input);*/
    let output_ids = llama.generate(input_ids, 500, 0.8, 30, 1.);
    println!("{}", tokenizer.decode(&output_ids, true).unwrap());

    println!("\n---------chatbot-------------");
    ChatManager::new(llama, tokenizer).run(); // 启动对话管理器
}
