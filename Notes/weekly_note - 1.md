# Week 1

## What Was Completed

**i) Replicated `micrograd.py`** — ([video](https://www.youtube.com/watch?v=VMj-3S1tku0))

Implemented the `Value` class, which enables mapping out a system of arithmetic operations on class objects. This allows us to calculate global gradients and pass them back to objects of previous operations along with the local gradients. The main work was writing a `backward_pass` function to propagate gradients properly. We also explored the `graphviz` library, obtaining a basic understanding and writing our own visualizer.

Using this class, we built:
- A `Neuron` class (an array of `Value` objects)
- A `Layer` class (an array of `Neurons`)
- An `MLP` class (an array of `Layers`)

We were successfully able to work on a toy dataset and write a training function for the MLP.

---

**ii) Replicated `makemore_bigram.py`** — ([video](https://www.youtube.com/watch?v=PaCmpygFfXo))

Using a [name dataset](https://github.com/karpathy/makemore/blob/master/names.txt), we set up a bigram prediction system. We compared two models:
- A **probability-based model** using bigram frequencies calculated directly from the dataset
- A **single-layer neural network** built using PyTorch

We implemented the dataset setup for prediction, the logic of logits, softmax for probabilities, and log-loss (essentially Cross-Entropy).

Comparing the two, the one-layer neural network achieved nearly the same performance as the known-probabilities model — demonstrating that a model can sufficiently mimic the properties of a dataset by learning from inputs.

---

**iii) Built our own Trained Generative Model** — ([video](https://www.youtube.com/watch?v=kCc8FmEb1nY))

Set up a dataset on Shakespeare's complete works, structured similarly to previous work: if `x_i`is a given character, `y_i` is the next character to predict.

Implemented the following:
- `Self-Attention` and `Multi-Head Attention` classes, with a focus on attention and causal masking
- A `FeedForward` class
- A `Block` class (following the *Attention Is All You Need* paper), consisting of residualconnections around two submodules: Multi-Head Attention and Feed-Forward
- A `Model` class wrapping everything — takes input, generates token + positional embeddings, passes through the blocks, and finally through a linear layer and softmax for token prediction

---

**iv) Overview of HuggingFace** — ([quicktour](https://huggingface.co/docs/transformers/quicktour))

Worked through the following pipeline using GPT-2:

1. Load the model (GPT-2) and its respective tokenizer
2. Use the tokenizer to obtain tokens
3. Push the tokenized inputs into the model to get outputs
4. Decode the generated tokens (using the tokenizer) to produce text

We also explored the model's configuration, its structural/computational layout, and gained anunderstanding of how `Pipeline` and `Trainer` work.

---

## What Was Learned

i) Gained an intuitive grasp of how operations are tracked in ML libraries like PyTorch, and how gradients move, not just theoretically or mathematically, but in terms of how it is executed incode (autograd).

ii) Understood why pre-order traversal does not work as reliably as post-order + reversal forsetting up an ordered list for back-propagation (due to the greedy nature of traversal; discussedin `confusion_log.md`, Page 1, Q1).

iii) Grasped the nuances of broadcasting, its evident utility for ease-of-use, but also how it can silently cause errors when the wrong dimensions are broadcast.

iv) While we already knew that normalization in attention ensures values tend toward a standard normal distribution (providing well-defined boundaries for values and aiding learning), we also understood that it prevents the softmax from becoming sharper and sparser for larger dimensions.

v) Understood why Feed-Forward is necessary and Self-Attention alone isn't sufficient:Self-Attention is essentially a weighted average of all tokens (where weights represent affinities between tokens). It acts as a communication layer — but that's about it. The Feed-Forward networkhelps in better extracting information from the processed token.

vi) Revisited Residual Connections and their necessity in deep networks: they primarily aid earlier layers in learning via back-propagation.

---

## What Blocked Us

In Assignment IV, we were asked to identify where residual connections occur in the model. Unfortunately, it wasn't possible to explicitly locate the residual connection (which is essentially an addition operation) through standard inspection. `Hooks` were explored as a potential solution, but at best they only expose the inputs and outputs of a particular layer which wasn't sufficient to isolate the residual addition specifically.