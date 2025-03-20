use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::slice::IndexOp;
use safetensors::tensor::TensorView;
use safetensors::{SafeTensors, View};

pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    pub rms_out_w: Tensor<T>,
    pub lm_head: Tensor<T>, // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        let get_tensor = |name: &str| {
            safetensor
                .iter()
                .find(|(tensor_name, _)| *tensor_name == name)
                .map(|(_, tensor)| {
                    let data = tensor
                        .data()
                        .chunks_exact(4)
                        .map(|chunk| {
                            let mut bytes = [0u8; 4];
                            for i in 0..chunk.len() {
                                bytes[chunk.len() - i - 1] = chunk[i];
                            }
                            f32::from_be_bytes(bytes)
                        })
                        .collect();

                    Tensor::<f32>::new(data, &tensor.shape().to_vec())
                })
        };

        LLamaParams {
            wq: vec![
                get_tensor("model.layers.0.self_attn.q_proj.weight").unwrap(),
                get_tensor("model.layers.1.self_attn.q_proj.weight").unwrap(),
            ],
            wk: vec![
                get_tensor("model.layers.0.self_attn.k_proj.weight").unwrap(),
                get_tensor("model.layers.1.self_attn.k_proj.weight").unwrap(),
            ],
            wv: vec![
                get_tensor("model.layers.0.self_attn.v_proj.weight").unwrap(),
                get_tensor("model.layers.1.self_attn.v_proj.weight").unwrap(),
            ],
            wo: vec![
                get_tensor("model.layers.0.self_attn.o_proj.weight").unwrap(),
                get_tensor("model.layers.1.self_attn.o_proj.weight").unwrap(),
            ],
            w_up: vec![
                get_tensor("model.layers.0.mlp.up_proj.weight").unwrap(),
                get_tensor("model.layers.1.mlp.up_proj.weight").unwrap(),
            ],
            w_gate: vec![
                get_tensor("model.layers.0.mlp.gate_proj.weight").unwrap(),
                get_tensor("model.layers.1.mlp.gate_proj.weight").unwrap(),
            ],
            w_down: vec![
                get_tensor("model.layers.0.mlp.down_proj.weight").unwrap(),
                get_tensor("model.layers.1.mlp.down_proj.weight").unwrap(),
            ],
            rms_att_w: vec![
                get_tensor("model.layers.0.input_layernorm.weight").unwrap(),
                get_tensor("model.layers.1.input_layernorm.weight").unwrap(),
            ],
            rms_ffn_w: vec![
                get_tensor("model.layers.0.post_attention_layernorm.weight").unwrap(),
                get_tensor("model.layers.1.post_attention_layernorm.weight").unwrap(),
            ],
            rms_out_w: get_tensor("model.norm.weight").unwrap(),
            lm_head: get_tensor("lm_head.weight").unwrap(),
            embedding_table: get_tensor("lm_head.weight").unwrap(),
        }
    }
}
