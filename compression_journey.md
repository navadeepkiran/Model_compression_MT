# Gemma 3 12B Compression Journey (WMT26)

## 1. The Objective

Our goal was to take the massive 12-Billion parameter `google/gemma-3-12b-it` model and heavily compress it so it could run, train, and generate fast English-to-Chinese translations within the strict 15GB VRAM limit of Kaggle's free T4 GPUs.

## 2. The Calibration Data

To figure out which parts of the model were important for translation, we couldn't just guess. We used exactly **1,000 translation pairs** from the `Helsinki-NLP/opus-100` (`en-zh` split) dataset. By running the model through these 1,000 English-to-Chinese examples, we forced the model to "light up" its translation-specific neural pathways.

## 3. The Initial Approach: Wanda (Unstructured Pruning)

We initially considered algorithms like Wanda and SparseGPT, which perform "unstructured pruning" (setting individual weights to zero).

* **Why we abandoned it:** Unstructured pruning doesn't actually make a model smaller or faster unless you write highly complex, custom sparse CUDA kernels (which are notoriously slow on T4 GPUs). Furthermore, we couldn't load the 24GB FP16 model to prune it without instantly crashing Kaggle, and you can't easily prune weights that are already locked into INT4 quantization blocks.

## 4. The Breakthrough: Fisher-Based Structured Pruning

We switched to **Structured Pruning**, which physically deletes entire layers and neurons, actively shrinking the architecture's dimensions.
To figure out *what* to delete without calculating gradients for all 12 Billion weights (which causes instant Out-Of-Memory errors), we used the **Fisher Information Masking** trick.

* We loaded the model in INT4.
* We attached tiny, trainable scalar "masks" (multipliers initialized to `1.0`) to the outputs of every Layer and every FFN Neuron.
* We passed the OPUS-100 data through and computed the gradient of the Translation Loss with respect to *only* the masks.
* Squaring these gradients gave us the **Fisher Information Score**, which perfectly answered the question: *"How sensitive is our translation accuracy to this specific layer/neuron?"*

## 5. Overcoming Hardware Limitations

Running this on Kaggle's dual 15GB T4 GPUs required massive PyTorch engineering to prevent crashes:

* **The 1.88 GiB OOM Fix:** We realized PyTorch was trying to allocate 1.88 GiB of VRAM just to store gradients for the unquantized 256,000-token vocabulary head. We fixed this by explicitly freezing the base model (`model.requires_grad_(False)`), speeding up the code 10x.
* **The CPU Offload:** We used `llm_int8_enable_fp32_cpu_offload=True` to dump the massive token embeddings into Kaggle's 30GB of System RAM, freeing up the GPUs for the transformer blocks.
* **Graph Fragmentation:** We initially tried to calculate Fisher scores for Layers, Attention Heads, and FFN Neurons all at the same time. The massive computational graph shattered the CUDA kernel. We solved this by isolating the runs (`MODE="layers"`) and dropping Attention Head pruning entirely (as Grouped Query Attention makes head pruning unstable).

## 6. The Hierarchical Slicing Optimization

Our final stroke of brilliance was making the pipeline **Hierarchical**.
Instead of calculating Layer scores and Neuron scores at the same time on the 48-layer model, we decided to:

1. Calculate Layer scores on the 48-layer model.
2. Physically slice out the 8 weakest layers (saving a 40-layer model).
3. Recalculate Neuron scores on the surviving 40-layer model.

* **Why this matters:** When you delete 8 layers, the flow of information shifts. By calculating neuron importance *after* the layers are dropped, we get a hyper-accurate map of which FFN neurons are actually useless in the shrunken model!

## 7. Next Steps

Once the model is fully sliced (Layers + Neurons), we will load the resulting lightweight architecture and run QLoRA fine-tuning on it to recover any lost accuracy and master the WMT26 translation task!


* **Layer 47** (Score: `0.000006`)
* **Layer 13** (Score: `0.1069`)
* **Layer 9** (Score: `0.1115`)
* **Layer 8** (Score: `0.1121`)
* **Layer 11** (Score: `0.1144`)
* **Layer 12** (Score: `0.1192`)
* **Layer 0** (Score: `0.1210`)
* **Layer 7** (Score: `0.1223`)
