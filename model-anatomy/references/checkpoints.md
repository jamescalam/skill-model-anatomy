# Checkpoints

## Read directly

| Given | Read |
|---|---|
| a folder with `model.safetensors` | that file |
| a folder with `model.safetensors.index.json` | every shard it names |
| a folder with other `*.safetensors` | all of them, as one model |
| a folder with `pytorch_model.bin` or its index | with torch |
| a folder with `.bin` `.pt` `.pth` `.ckpt` only | with torch |
| one `.safetensors` file | that file |
| one PyTorch file | with torch, `weights_only=True`; a dict under `state_dict`, `model` or `module` is unwrapped |

safetensors files are memory-mapped: only the header and a strided sample of each tensor are read, so the model
never has to fit in memory. PyTorch files are loaded whole on the CPU.

dtypes: F32, F16, BF16, F64, I64, I32, I16, I8, U8, BOOL. FP8 tensors are drawn at their true size with flat values.

## Not read: convert first

| Format | Convert with |
|---|---|
| GGUF (llama.cpp) | use the original Hugging Face safetensors instead; GGUF weights are quantised blocks |
| ONNX | `onnx` + `numpy_helper.to_array` for each initializer, then `safetensors.numpy.save_file` |
| Keras `.h5` / `.keras` | load in Keras, then `safetensors.numpy.save_file({w.path: w.numpy() for w in model.weights}, …)` |
| Flax `.msgpack` | `transformers`' `from_pretrained(..., from_flax=True)` then `save_pretrained(safe_serialization=True)` |
| TensorFlow SavedModel | `from_pretrained(..., from_tf=True)` then `save_pretrained(safe_serialization=True)` |

Any of them stops the build with a line saying what it found and to convert it. Never upload a model to a converter
service: convert locally.

## Quantised checkpoints

GPTQ/AWQ/bitsandbytes files store packed integers (`qweight`, `scales`, `zeros`): they draw at their packed sizes, which
is not the model's real shape. Use the unquantised checkpoint.
