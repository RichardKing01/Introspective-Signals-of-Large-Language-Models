# Confusion Log

---

## Day 1

It quite nice to see Mr. Karpathy mentioning about the equation on finding a differential at a given point. It is good to notice one interesting aspect which is that f(h+x) – f(x) is indeed the 'sensitivity' of a function. Now, dividing it by 'h' or normalizing it by h tell us 'sensitivity'/over unit change, this is your slope, or direction (positive/negative) and strength (magnitude) of change.

Now, Mr. Karpathy glosses over the fact as to why he's used tuple datatype for the children attribute. Instead he can use the rather comfortable list. In Python lists are not fixed, they are extendable and actually have quite the overhead, being ordered and having multiple class_operations. But in tuples, tuples are quite efficient in being fixed size indexed, have a very limited set of functionality, and not being extendable, is quite efficient.

It is quite important to notice the point that backward isn't the slope of a given variable, but rather it is the slope that it sends back to the variables it is dependent on.

Another important notion that brings the user's attention was until this point of time, each variable was contributing to a single (next) variable. But this isn't the case in a Standard MLP. Now, given that this is the case, in the original backward propagation, the code had done this:

```
=> dL/dw = dL/dh * Dh/dw
```

Now if there were two variables, h1 and h2, the above update only takes the gradient from one variable, and that wouldn't work! 😞 Hence, the Author guides us to accumulate the gradient, which basically means to sum the gradients coming from various sources rather than a single update:

```
=> dL/dw = (dL/dh1 * dh1/dw) + (dL/dh2 * dh2/dw)
```

**Q1:** Why didn't my pre-order method for building out the chain didn't work for all cases? Why did the post-order + reversal work?

From my understanding, there is a chance of failure, especially when two variables have the same dependent variable (variable they depend on). What happens in the post-order search (left-right-root), is that by the very notion the root comes at the end, so when we reverse it, the root or rather roots come at the end, making it very easy to back-propagate.

Now, in the pre-order stance, it would work nearly all the time, given it's a 'reverse' of the post-order method. But the problem lies in the fact that, as the algorithm traverses greedily, if two variables have the same dependent variable, the algorithm passes through one of these variables, and explores the dependent variable without registering the other variable, this would cause issues in the back-propagation, where only one of the two variables properly update the dependent variable.

The Loop seems to be the prima aspect of the MLP portion of the study. It states the following steps:

i. Forward Pass
ii. Zero the Gradients
iii. Backward Pass
iv. Update the weights

**What would happen if you called it twice without resetting gradients?**

Points (i), (iii) and (iv) make some intuitive sense, but zero the gradient or 'zero-grad' doesn't really do so. But we have to recall the fact that we at any given time accumulate the gradients (+=). Now for every iteration or 'epoch', if one does not clear off the gradients, the past gradient is added to the current gradient, biasing the updates and thus directing the update in a wrong decision. Also, as we accumulate the gradients, the gradients might make the updates or the updated instances go out of bounds!

**What does `Value.backward()` do, in your own words?**

Value.backward() is two parts: As we've previously stated, relays the 'global-gradient' that it (a certain instance) has accumulated along with its own slope w.r.t. to its parent instances, to its parent instances, this is what Value._backward() in the nuclear sense does. Now, slope of gradient is passed through all the chain, from the final node or instance, to the beginning instances (and in that particular order, child nodes first, then parent nodes).

---

## Day 2

**Write Note:** tend to be quite deliberate with what you're looking and doing that, write down the thought process in your understanding of the concepts and frameworks.

**Python-note:** zip halts if unequal lengths, and returns the List of tuples of the same length as the shortest length.

**Q) Why exactly is Multinomial or rather `torch.multinomial()`?**

In the video, we don't get a major understanding of what exactly is `torch.multinomial()`. Exploring the [documentation](https://docs.pytorch.org/docs/main/generated/torch.multinomial.html), we find that it is rather based on the Multinomial Distribution. What exactly is it? The best way to say it, it is the broader description or a generalisation of the Binomial distribution, where instead of studying a Binary System with n-trails, we study a m-system n-trails. Note that this is a very generalized view of this. Sum of all events must be 1. *This isn't necessarily important for our study.*

Now, why is this linked to this particular project? Well, you can consider that, thus far you consider the count of each character-bigram in your dataset, normalizing them with the total no. of instances leads us to the probability of each bi-gram occurring in our dataset. Now, `torch.multinomial()` is necessarily a sampling function, wherein it samples a set of indices (or rather for us a given bi-gram), based on its probability, higher the probability, more likely are you to see it being sampled.

**Q) What is each aspect of 'fixing' probability is in torch?**

First, we need to understand the fact that there is no non-determinism with a deterministic machine such as computers. Everything is pseudo-deterministic, meaning that they seem to be random, but are actually deterministic produced by a certain algorithm. This is where Pseudorandom Number Generator is, based on a certain input or 'seed', it produces a long list of random-looking sequences of samples/outputs/numbers.

`torch.Generator` is exactly that Pseudorandom Number Generator and we set up a seed using `.manual_seed(int)`. Now the values produced from this Generator is a float ∈[0,1). This is then mapped to the certain distribution that we follow which is basically f(x).

**Q) Clearly Broadcasting seems to be important… how does it work?**

We are able to understand that dissimilarly sized matrices or tensors are able to do some sort of arithmetic by doing some sort of duplication along certain axis. In the [Torch documentation](https://docs.pytorch.org/docs/2.11/notes/broadcasting.html#broadcasting-semantics), we find that torch actually follows through with Numpy's Broadcasting, which is based on two simple rules, interestingly enough.

Arithmetic between two matrices or two objects are followed through if as we move from right to left in terms of Dimension:

i. All Dimensions are equal
ii. **ONE** dimension is 1

If it is one, duplication along this dimension occurs to process the arithmetic!

**Bug) Why keepdim argument was necessary for the sum?**

Noting the nuances of Broadcasting, the duplication of the values would certainly be different based on the keepdim argument; consider the following example.

DF is of the size (27, 27).
Now consider `DF_sum = DF.sum(dim = 1, keepdim = True)` => `DF_sum.size() = (27, 1)`

If we were to broadcast:
```
DF           | 27, 27
DF_sum  | 27,  1
Effective | 27, 27
```

But if we were to not consider the keepdim:
`DF_sum.size() = (27)`

If we were to broadcast:
```
DF           | 27, 27
DF_sum  |       27
Effective | 27, 27
```

So in the original, the values were copied to column-size, meaning values of the row were copied along the column, whereas in the other one, the values were copies to row-size, meaning the values of the column, were copied along the rows.

**This is quite challenging as it doesn't shoot any error or warning, but it would lead to a lot of time being wasted.**

> *Note: This model that has been made thus far (1:04:30) is actually a basic prediction system, based on the current character, what's the next best character… it is an iterative 'greedy' approach!*

**Note: Some Quick Observations or notes for understanding the Forward Propagation in MLPs.**

```
W = (1 x n) => One Neuron
W_m = (m x n) => m neurons

X = (n x 1) => One input
X_k = (n x k) => k inputs

Now then:
Out = W @ X = (1x1) (one output for one neuron)
Out_m = W_m @ X = (mx1) (one output for m neurons)
Out_m_k = W_m @ X_k (mxk) (k outputs (one for each input) for m neurons)
```

> *Something to notice is that these are reversed in Torch, the above is for our understanding following the slope-intercept formula: y = mx + b*

**Q) Making sense of the logits, counts and probability nomenclature on Deep Learning**

Initially, this don't make much sense, but we sort of have to walk backwards to make sense of what's going on.

Recall the fact that the probability that a given word is 'valid' is the multiplication of the probabilities of its constituent characters' probabilities. Now, these values are so small, so we resort to a monotonic function, the log, so that it's easier to calculate, wherein, instead of multiplying the probabilities we add the **log-probabilities**.

So basically:
```
log-probabilities = loge(probabilities)
=> e^(log-probabilities) = probabilities
```

Now, we take a similar rendition for Neural Training, where we say, let **X @ W** be log-counts or logits. Then exponentiating them would give us counts (e^log(counts) = counts). Then we know the probability for a given character is basically Counts/sum(Counts) which is effectively e^log(counts)/sum(e^log(counts)) which is a softmax function, giving us the probabilities.

*Exactly how did we come to the conclusion of logits, even I'm unsure at this point of time, but it really isn't a problem at this point of time.*

**Note:** The `tensor.item()` is to remove the torch wrapper over the python item (int or float) to provision the python class-object.

We are able to see that the very Simple model (A single Layer of Neurons) is able to neatly replicate the behaviour of the Dataset. This can be seen by comparing the Probability based system's overall loss and comparing the Neural Network's Loss, both of which converging to about the same mark with some minor deviation.

With this example we can convincingly say that the neural model has learned the same transition behaviour as the actual dataset, effectively modeling the same distribution. Interestingly enough, if you recall our W (being the 27x27 matrix), it effectively is the log of the counts.

**Q) Why log?**

log is a monotonic function (singular direction), and is a smoothening function to a set of discrete values. This makes things easier for a Machine to work on, as Neural Networks work on precise floating-point arithmetic anyways.

**Primary Common Bug:** Accessing Tensor data, understanding how to properly include the index that we get the right data.

**Summary:**

i. How we've come about setting up logits, and then using exponents to obtain the counts, the motivation behind it is something that still confuses me. But we accept the fact that softmax's are differentiable and thus, back-propagate(able), which is quite useful for Neural Networks.
ii. `torch.Tensor(x)` and `torch.Tensor([x])` are inherently different things, wherein one is a scalar and the other is a 1x1 matrix, this caused a ten-minute meltdown.
iii. Broadcasting took us some time to get through, not conceptually, but in practice, especially with the aspect of dimensions that what dimensions are being broadcasted (that has been detailed in the confusion_log.md)
iv. We had a small bug of not removing the tensor wrapper (`tensor.item()`), which did break slightly.

Other than these, all things are quite intuitive and there weren't really any major breakdown.

---

## Day 3

**Note:** A constant struggle, which requires constant attention especially as we're trying to replicate programmes on one's own, is keeping track of dimensions continuously, especially as we're multiplying and any operations.

**Q) What in God's beautiful world is `self.register_buffer`?**

"PyTorch is a method used within nn.Module classes to register a tensor that is part of the module's state but **should not be considered a learnable parameter**."

**Q) In attention why normalize the scores with √d?**

In attention, we normalize the scores by √d for two reasons. First, it controls the scale of the dot products so that their variance doesn't grow with the vector dimension, keeping the distribution of values roughly stable. Second, without this scaling, the softmax tends to produce very sharp, almost one-hot distributions as the dot products grow, which can lead to a loss of useful information. Dividing by √d prevents this and ensures that the attention mechanism remains smooth and stable during training.

**Q) Why `nn.ModuleList` instead of just ordinary lists?**

It is very true that `nn.ModuleList` is the PyTorch equivalent of a standard list, but instead of holding normal datatypes such as int or float (or tensors for that matter), `nn.ModuleList` is used to hold objects of neural-network functionality. This is necessary because a plain Python list doesn't register its contents with the parent module, so those weights won't show up when inquiring the wrapper class-object via `model.parameters()` or `model.state_dict()`. Using `nn.ModuleList` ensures they are properly registered and visible. This also means the model behaves correctly as a whole when saving/loading checkpoints, moving to devices with `.to()`, switching between `.train()` and `.eval()` modes, and so on.

Note that backpropagation works either way: autograd traces the computation graph during the forward pass and doesn't care about the container. The real danger with a plain list is that the optimizer silently won't update those weights, since it only sees parameters returned by `model.parameters()`.

Basically, `nn.ModuleList` is about ownership and discoverability: it's for proper registration, visibility, and correctly utilising the model as a whole.

**Q) Why `super().__init__()`?**

If we notice, whenever we initialize a torch neural-network or computation class, it is written as `class ClassName(nn.Module)`. This means that our class is the child class of `nn.Module` (Inheritance from OOPs). If we intend to inherit its internal machinations, one needs to initialize the parent class as well, which is what `super().__init__()` does!

**Q) What was the necessity of the Feed-Forward Neural Network?**

In the paper, 'Attention is all you need' by Ashish et. al., it mentions about the Feed-Forward Neural Network in passing. But Mr. Andrej makes a very interesting point about why Feed-Forward (basically an MLP) was necessary.

If we consider the Attention Mechanism in our Decoder-Only Model (which is basically what all Generative Language Models are), it is a very fancy way of doing weighted average of the various available tokens. Andrej describes it as it is aggregating from the neighbouring tokens or "tokens talking to each other". But this is about it, with the addition of the context we have seen some reduction in loss and better generation. But since these are only communicating, without any learning or this aggregated information, a FF-NN is employed, this will aid us in extracting some information of this data, to produce some good results, especially when non-linearity is involved.

**Q) Why `dim = -1`?**

We have to notice that as we go into more complex systems, the data that is flowing can be 3 or even 4-dimension tensors. In such a case, when we wanted to in particular sum or normalize or do some process along some directions; the `-1` just notifies the last dimension which usually in our case is the vector representation of the token.

**Q) The necessity of embedding the token.**

*(See architecture_notes.md)*

---

### Summarizing the Final Setup

**i) Data Loading:** We load in the Data into two separate parts, X and Y. Where if X = Sentence[ti: tf], Y = Sentence[ti+1:tf+1], this is so that each xi is responsible to predict the next word in the sentence, which would be yi.

**ii) Self-Attention Head:** The Self-Attention head takes in the **token-embedding**, generates the respective key, value and query vectors for each of them using the Key, Query and Value weight matrices. Now, we move forward to generate the scores, done using the formula of attention: `score = softmax{(q ∙ k)/√d}`, and using this for the weighted average, which would be the 'attended' token. It is very important to notice that we actually mask or set to 0 the weights for future tokens. This is necessary, because we intend to predict future tokens based on the currently known tokens — acknowledging the future tokens in this manner would lead to 'data leakage'.

**iii) Multi-Head Attention:** Basically initializing multiple heads, concatenating the outcomes of the heads, passing it through a projection layer, to obtain the final 'attended' token.

**iv) Feed-Forward Neural Network:** The part which takes in the 'attended' tokens, and extracts some information by passing it through the network, which includes some non-linearity functions such as Dropout and ReLU and passes it out.

**v) Block:** This part groups together the submodules, in particular the Multi-Head Attention and the Feed-Forward Neural Network submodules. It is very important to notice that here, the Residual connections are implemented. Another note is that in the paper 'Attention is all you need', add and normalization was done usually after a given sub-process, but a modification done was doing it (Normalization) before a sub-process, for better learning and better back-propagation.

The Block does the following steps:
- a) Multi-Head Attention of the Layer-Normalized input.
- b) Add the original input to the above outcome.
- c) Feed Forward Neural Network of the Layer-Normalized intermediary b.
- d) Add the intermediary b to the above outcome and return.

**vi) Model (written BigramLanguageModel):** This is our Decoder System. It consists of an Embedding Table, that takes in our sparse input-token and produces a larger dense embedding. It also consists of the position_embedding, taking in the particular positions of the token to give a position-embed vector. A Linear Layer is present to convert the internal representations into the final logits. A set of Blocks exist, where each block consists of the above.

The Model's **Forward pass** is as follows:
- a) Generate the embedding for a given token.
- b) Generate the positional embedding, provided the position of the token.
- c) Add the two vectors to produce an internal representation.
- d) Pass this through the multiple Blocks.
- e) Finally, pass it through the Linear Layer which produces the Logits.

The **Backward Pass** is as follows:
- a) Calculate the loss using Cross-Entropy.
- b) Zero the gradients of weights.
- c) Loss-Backwards (calculate the gradients).
- d) Update the weights using an Optimizer (AdamW).

The Model **generates** by the following:
- a) Initially taken in an Input sentence.
- b) Takes in the last-n tokens, based on the sentence-block.
- c) Pass it through the Forward pass to obtain Logits.
- d) Calculate the probabilities using the Soft-max.
- e) Select the next character based on these probabilities (sampling).
- f) Appending this character to the sentence and passing it through again.

**Q1) Why is attention "all you need" — what does it accomplish that an MLP alone can't?**

When using the MLP layer, we find that a model generates based only on the current character/word/sub-word. Attention, as described, is a communication layer, which sees the available token, and produces new representations for each of the tokens, which is a weighted average of the available tokens. This now produces a token which is quite-rich in information and is a context-based representation of a given token.

**Q2) What is a residual connection, and why does the GPT block put one around both attention and the MLP?**

Putting it very plainly, a residual connection does the following: `out = x + f(x)`. We have to understand that, especially for deep models, the computing sub-modules initially produce quite garbage outputs. To train these models effectively, the gradients don't carry much magnitude as they travel back to the earlier layers — this is the vanishing gradient problem.

Thus we introduce the residual connection, which contributes to the outcome while making sure that better backpropagation and healthier gradients get to the earlier layers as the x part of out, creates a direct highway for gradients to flow straight through, bypassing the submodule entirely.

**Q3) What does causal masking do? And why is it needed for autoregressive generation?**

It is very important to notice that we actually mask or set to 0 the weights for future tokens. This is necessary, because we intend to predict future tokens based on the currently known tokens — acknowledging the future tokens in this manner would lead to 'data leakage', effectively allowing the current token(s) to know about the future tokens, which isn't proper prediction or learning properly prediction.

**Additional Resources:**
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Attention Is All You Need (paper)](https://arxiv.org/pdf/1706.03762)
- [Deep Residual Learning (paper)](https://arxiv.org/pdf/1512.03385) *(Introduction in particular)*

---

## Day 4

**Q) Briefly explain about the Byte-Pair Encoding**

i. Load the Model
ii. Use the Tokenizer to get the Tokens
iii. Push it into the model to get the respective outcomes
iv. Decode the generated tokens to produce a text

**Q1) What is the shape of `model(**inputs).logits`? Why those dimensions?**

The shape of the logits were tested as follows:

```python
with torch.no_grad():
    outputs = model(**model_inputs)

B, T, C = (outputs.logits.shape)

print(f'Batch size : {B} | No. of Token given as input: {T} | Size of vector representing Processed Token: {C}')
```

Output: `Batch size : 1 | No. of Token given as input: 9 | Size of vector representing Processed Token: 50257`

So originally we had given about a string of about 36 characters (including spaces), and GPT2's tokenizer produced 9 tokens out of this. We have to remember the fact that GPT2's tokenization process follows the Byte-Pair Encoding method.

> *Small Note on Byte-Pair Encoding: It is the process of creating sub-words by greedily concatenating characters based on their frequency in the dataset.*

Since we gave one string as input, wherein the string had become into a group of 9 tokens. Now according to the GPT2's tokenization, the vocabulary of the GPT's system has about 50,257 sub-words that have been generated from Byte-Pair encoding.

This is further instantiated by the following:
```
(wte): Embedding(50257, 768)
(wpe): Embedding(1024, 768)
```

The block size is 1024 in the GPT-2 model. The vocabulary size is 50,257.

**Q) Where in the model object would you find the residual-stream activations?**

From looking through the configuration of the sub-modules, we find that a Block (part of a set of blocks: `model.transformer.h`) has them, primarily identified by the `resid_dropout` attribute, which is a dropout after the residual process (`out = x + f(x)`) is done. This is accepted because in a (standard) transformer block, the residual connections are implemented, one after an Attention function and the other one after the FFNN.

**Q) How do `attention_mask` and `position_ids` differ in role?**

**`attention_mask`:** Attention_mask is quite important when we consider a batch input. When we give a set of sentences, there is no particular guarantee that the length of these sentences are the same, and even if they are there is no guarantee that the tokenizer tokenizes the sentences such that the processed sentences are all of equal length.

Since a model requires that in a given batch, the length of the blocks in the batches have to be equal; the tokenizer pads the sentences to the maximum-length block in the batch. Along with this, it also generates an `attention_mask` to each of these blocks so that when the model generates or passes it through the forward, it ensures that it does not accommodate the padding-tokens, but only the actual tokens, hence 0 for padding and 1 for genuine.

Going even deeper, during attention, the following occurs:
```python
mask = (1 - attention_mask) * -1e9
scores = scores + mask
```

In this way, the Padding vector don't contribute to the prediction, and it is hence why `attention_mask` is quite important.

**`position_ids`:** Another important value, it informs the model the particular position of the token. As you can recall, in Transformers, Attention mechanisms and even the FFNN are unaware of the position — that is, without us explicitly provisioning some information regarding the position, these mechanisms wouldn't be able to best predict the next tokens. Thus we provide along with the `embed_token`, the `position_ids` which are values that help each token to obtain its respective positional-embedding from the **(wpe): Embedding(1024, 768)**.

---

### Understanding Pipeline and Trainer

**Pipeline:** Basically, a wrapper so that we are able to inference a model to obtain the necessary outcome considering the task given as input:

```python
Pipeline(task, model, device)
# task: can be 'text-generation', 'image-segmentation' etc.
# model: A particular model name can be provided, but the Pipeline would automatically set a model.
# device: set to some default value, and can be set by user
```

So now, to infer the model, one can just do `Pipeline(Query)` to obtain the output.

**Trainer:** Quite similar to Pipeline, but instead it allows for an easy interface with these models, for training them.

It primarily consists of one main aspect, which is the `TrainingArguments` class, this helps us tell a couple of things like:

i. `output_dir`
ii. `evaluation_strategy`
iii. `learning_rate`
iv. `per_device_train_batch_size`
v. `per_device_eval_batch_size`
vi. `num_train_epochs`

This is then fed to the Trainer Class:

```python
trainer = Trainer(
    model=model,
    args=TrainingArguments_object,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
```

To train, one just does `trainer.train()`. To access the trained model, we can just do `trainer.model()`.

---

## Week 2

A very interesting comment that was made by the ARENA in their introduction to Transformers was that the Attention mechanism, being the communication layer, is also a simplified convolution, but instead of a local view of the data (as done in CNN), it looks through all the units of the given data. This might be very relevant to the understanding or interpretation of Transformer Circuitry.

Though obvious, it has to be mentioned here, that MLPs play the role of learning logic, it's the logic store of the block, and in that manner, the final MLP player plays the global-level of logic store. Mapping a given embedding to a vector of logits.

**Comment: Why addition is just as powerful as concatenation.**

Recall that during (Week-1, Day-1), the addition arithmetic equally splits the gradient that is being passed back. From a learning perspective, this is quite useful and simpler than concatenating and passing through some MLP-Layer which increases computation complexity and also increases the number of weights.

Additionally, if you consider the residual stream for a second, we are able to compress quite the information into a contained vector shape. The machine, especially through backpropagation is able to see a + b + c… + n distinctly, and so is able to learn from each contribution. Consequently, future layers can extract and leverage each component effectively, despite the superposition, achieving comparable representational power to concatenation without the extra overhead.

**Comment: On sampling from a LLM**

Recall our conversation regarding Multinomial (Week-1, Day-2), it was about the generation of samples for selection based on the probabilities that are obtained. But we forget to mention the manner by which we use the Multinomial tool, which was a greedy method: For a given input, find the next probable token and repeat.

But there are other ways:

i. **Temperature:** when calculating the probabilities from logits, let the equation be: `exp(xi/θ) / Σ exp(xj/θ)`. This is very similar to the Softmax that we're used to, but with the addition of a Theta as a denominator to the power. Larger a theta value, smoother are the distributions, and smaller the value, sharper the distributions. Using this, one can use the greedy approach or do the following.

ii. **Beam Search:** At any given time, select the best k-valued instances.

iii. **Top-p Search:** Since the outcomes can be seen as probabilities (and are probabilities), select the smallest n-set of tokens (or group of tokens), which have a combined probability of less than p%. This way one can get a diverse set of strings.

iv. **Top-k Search:** At every iteration, select the top-k items based on probability, and normalize their probabilities for future comparison.

**Computational Necessity of Caching:** The Concept of Caching is a subtle efficiency technique. To boil it down, especially for Attention, it just says for the previous tokens, save the Keys & Values, such that one needs to compute the Query, Key and Value for the new token instead for all the tokens.

**What are In-Context learning heads or copying heads?**

This is something that has been studied to a decent extent in the study of mechanistic interpretability, particularly as well because we're limited by the understanding of Multi-Layer Perceptrons.

When we're only working on a single layer Attention, we find that Attention heads slowly equip themselves with a pseudo-type of copying in various different patterns. We identify that especially in one-layer models, the head dedicates a lot of their logic-store for this 'copying'. When we mean copying, the process is quite simple: ***The QK circuit is able to attend back to tokens which could plausibly be the next token. Thus, tokens are copied, but only to places where bigram-ish statistics make them seem plausible.***

In simple words, if a sequence has previously occurred, the QK circuit attends in such a manner that if a part of the sequence had occurred, it will try to copy and fill up the sequence.

***Some of these patterns of copying are:***
- i) [A][B]… [A][**B**]
- ii) [AB]... [A][**B**]

The most interesting one is (ii). The example provided in (A mathematical Framework for transformer circuits by Elhage et. al.) shows that given the previous word **Ralph** in the string, even if the tokens are broken down, since the tokenizer in modern Generative models aren't character-wise but are Byte Pair encodings… it still ends up trying to predict the full word even if in parts…

`[RALPH]… [R] [ALPH]`

This isn't rudimentary in my opinion, but there is some cognitive effort beyond just copying.

**Basic Patterns in Head functionality:** prev_token_heads, current_token_heads, first_token_heads

i. **Prev_Token_heads:** A very important functionality for induction-heads: The QK circuit provides the most importance to the token right before the token of interest.

ii. **Current_token_heads:** The QK circuit places the most weight to the destination token.

iii. **First_token_heads:** The QK circuit places the most weight to the first token of the sentence, which usually is a Special token (`<>`). This is done because one has to understand the sum of weights equate to one — when the system tends to not find any similarities, it places the weights to a term that is irrelevant.

**The Abstraction into QK Circuits and OV Circuits**

When we go through the attention mechanism, it strikes to us that there is primarily two processes that occur:
i. Obtaining the Score
ii. Weighted Sum of tokens (Mixing)

We also notice that these two processes are independent and are done by individual independent units, (i) being done by **Q** and **K** and the other being done by **V** and to the extent **O**. Thus we find that we can abstract the Attention mechanism to two parts, one the QK circuit being responsible for the scoring and the OV circuit responsible for the output of the Attention mechanism.

Interestingly enough, one can in fact only train two matrices instead of 4 and still obtain the same output.

**The Q-composition and K-Composition**

We are well aware of the point that in an attention head, a given vector representation of a token is abstracted into three different vectors, which are the query, key, and value. Now in-attention there have been a notice of a certain behaviour which I would like to coin as token-point-of-view & neighbour-point-of-view.

Now, when we're calculating the raw scores, one does the following: `QK^T`. An example can be seen below:

`[[q1k1 q1k2] [q2k1 q2k2]]`

If we read row-wise, we are quite clear in understanding how q1 notices each key of the dataset, and the columns read as the diverse reach of a key with respect to the queries — these notions are respectively part of the Q-composition and K-composition.

Q-composition tells the story of the query being the primary component of activity, wherein it attends to a diverse set of keys, this is quite uncommon in Attention behaviour. Whereas in K-composition, we find that keys play a primary component of activity, where keys are diversely attended to queries.

A key indicator of whether something is Q-composition or K-composition, is basically the reach of attention. For a given Query, if it looks like the Query is attended by seemingly obscure keys, or further keys, it is a Q-composition. If K-composition, the Attention is limited to the locality of the Query.

Now one may question why such a classification is required. But it stems from the point that it is indeed true that for a given classification task (which is basically what prediction is), one requires the local information (K-composition). If an Attention head is acting beyond what is reasonably local, one must understand it has learnt to track repeated entities, long-range dependencies, and other intrinsic properties.

Source: [Induction Heads Illustrated](https://www.lesswrong.com/posts/TvrfY4c9eaGLeyDkE/induction-heads-illustrated)

**What is a Hook?**

The Hook is a mighty tool that is used in Mechanistic Interpretability work. A hook is basically a certain position in the large system of the language model, where one accesses to see what the activation has entered/passed, and do modifications at that position to see the changes that occur. This is best introduced by the TransformerLens python library.

**What is an induction score?**

Given a set of sequence: [A, B, C, D, E, F] be a sequence that we intend to mimic with an attachment of A, such that the sentence looks like [A, B, C, D, E, F, A2]. Now induction score looks like the following:

For a given (induction) head, working on a token (which in our case would be A2), and see how strongly it attends to the Key of a token `sequence_length - 1` away (which would be B). We do this for multiple similar sequences, and we average it, producing the induction score for that given head. 😄

**What is Ablation?**

Similar in spirit to dropout, it renders certain outcomes or activations of neurons, heads, blocks and so on to zero to see the effect of whatever we are interested in (loss, accuracy, etc). This change can be done using hook functions in transformer-lens as we know that hooks can be used to modify the outcomes of the current layer.

**What is Direct Logit Attribution?**

Recall in (Week-1, Day-3), we had a look into Residual Networks. Notice a very important point that the residual connection or fire is basically as follows:
```
x' (residual) = x + f(x)
x'' = x' + f(x') => x'' = x + f(x) + f({x + f(x)})…
```

In this manner, we are able to push the contribution of each function or computing mechanism in the system all the way to the output.

So if that is the case, then it is quite easy for us to see the following: Given WU be the Un-Embedding (the MLP Layer that expands the output of the transformer into vocab_size), then `WU(x'') = WUx + WUf(x) + WUf({x + f(x)})`, since this is a Linear function. Now, in this manner, we are able to best see the contributions of each mechanism in the system to the logit.

**What are Induction circuits?**

A very powerful notion of logic that was identified for heads. We find that the ideation of 'copying' that was identified previously in one-layer, but these were slightly beyond statistical, having no apparent robustness to it.

Induction is a more interesting form, having understood the relational structure in sequences, looking into not what just occurred, but what also followed through for the predictions.

Induction heads can only occur in situations of more than two layers, i.e., the number of heads have to be at least two, and the outputs of one head is the input of the other. This aspect allows for the extension of the definitions of Compositions:

i. **Q-Composition:** The subspace of the output of the previous head affects the generation of queries.
ii. **K-Composition:** The subspace of the output of the previous head affects the generation of keys.
iii. **V-Composition:** The subspace of the output of the previous head affects the generation of values, which is a very interesting notion — one can think of this system as only a single head or one virtual head.

**On reverse-engineering Induction circuits:**

Up until now, what we have done is understand that a given pattern occurs in a (induction)-head. But we haven't come to the notion of trying to understand how it mathematically occurs, and whether it was a fluke for what we've seen. Below are some ways that are considered to work on understanding heads:

Before we go quite further in detail, we have to understand that induction heads show up when there exists two layers or rather two heads. One head exists as a prev_token_head, where for a given token, the head attends to it the immediate-previous token primarily.

These now attended tokens are passed onto the next attention-head. It gets to us now a question which is why prev_token_head is quite important for the induction head.

**Why was the prev_token_head necessary for the next layer's head to be an inductive one.**

If we consider [A, B, C, D, E, F, A2]. We want to say that, since clearly it looks like a repeating sequence, after A2 must be B2. But this works in a slightly interesting way — recall that A1 and A2 will be having a high score simply on the basis that they are the same vector. If we consider that this sequence passes through the prev_token_head, the token of `B = BE + f(A1) + pos(1)`. Now, notice that A1 is also a contributor to B's token… hence, when we do the dot product for the Key-Value, we are quick to identify that B will tend to be higher relative to the rest of the tokens. This along with the supervised learning that the next token will be B, will reinforce the weights in the head to prioritize bringing up the logit of B, given that B exists in `sequence_len – 1`, and that it is preceded by an A.

**Reverse Engineering the Attention Head or Understanding Diagnosing the type of Head**

**i) OV Copying Circuit:** When we intend to test our Hypothesis for whether a given Head is an inductive head, we do the following.

Consider a sequence [A, B, C, D, E, F, A2]. We test for the following: `B^T WE WO WV WU`. We see for the given head's OV Circuit, whether B would ever be a contributing factor to the logit of B (in the output sequence). If it is an inductive head, we find that to be the case.

**ii) QK Circuit:** Before, we considered inductive head, but now we consider how to identify whether a given head is a prev_token_head. If a head gives importance to position, it would mean that for a given vector representation, the position_encode will be the major contributor to things.

This would mean that if we do the following: `Wpos WQ WK^T Wpos^T`. If we would plot or view the matrix of this matrix multiplication, if we are able to easily identify a strong 'shifted-diagonal' matrix, then it indeed is a prev_token_head.

**iii) K-Composition Circuits:**

It is really similar to the notion when we've previously talked about this. At any time a given token x is equivalent to `xemb + pos + Σ(ai * yi)`, where the summation is the weighted average or the function of some attention per-se. Note that this is a generalization, and we can omit the summation if indeed it is an early stage.

Now we just intend to find the Key-value and Query value for the given `Wx = Wxemb + Wpos + Σ(ai * Wyi)`. In this manner we have decomposed the key/query value of x into its contributing elements. Now, once we've generated the decomposed key and query, there are two methods of comparison to determine whether something is K-Composition or Q-Composition:

i. **Normalize:** For each vector generated from the decomposition of the Query/Key value, normalize it to obtain the magnitude. The Magnitude gives us a vague notion of whether Query or Key plays a major role in the Attention Mechanism.

ii. For a given x, `qx = WQ x`, similarly `kx = WK x`. Now, then, scores are given by `qx * ky = WQ x (WK y)^T`. Now then, `Wx = Wxemb + Wpos + Σ(ai * Wyi)` and similarly would be y… Thus `score_ij = Σ_i Σ_j yi Wq ∙ (yj WK)^T`. From here, we get a sum of pairs of multiplication of components of queries and keys. From here, it's a good old comparison.

**iv) On understanding Composition Score:**

Without going too deep into the details, another way to determine whether a given system is an Induction Circuit, is to see whether the Output of a given system aligns with the Input of a given system.

Simplifying it all, if we consider `A = (WO WV)_1` & `B = (WO WV)_2`, we define the score to be a Cosine-Inspired Equation: `‖A · B‖_F / (‖A‖_F · ‖B‖_F)` where `‖X‖_F = √(Σ_ij W_ij²)`.

**Understanding Virtual Weight:**

Notion of subspaces being field, and residual stream acting as a memory.

**Abstraction of Outcome Portion of Attention:** It is necessarily a `W_i x_i` movement. Let us take the example: Given that `WO = (n_heads * d_model, d_model)` and the resultant vectors for the multi-head being `Z = (n, n_heads * d_model)`.

Now, notice one very important thing, which is that we can abstract `WO` as a matrix of Output matrices for each of the heads, and similarly the vectors (row-wise) can be abstracted as a vector output from each of the heads. This brings the very useful view of matrix multiplication, which would be Kronecker's multiplication.

**v) On identifying copying mechanism in single-layer Attention:**

One interesting way of identifying 'copying' mechanism in Attention is using eigenvectors. Consider a vector v which can be a summation of multiple vectors of similar properties (linguistically). If `M` defined as `WU Wh OV WE`, the v can be such that `Mv = λv`. Given λ is positive, we can notion that v is a group of tokens that mutually increase their own probability.

Now there are problems to this. For a 'random' matrix, the eigenvectors are not orthogonal, meaning they are not in subspaces mutually exclusive to themselves — leading to certain vectors being similar in direction. If we thus consider v as a singular vector, it can be decomposed into two components.

The problem is this: consider eigenvectors in similar directions. If we decompose a given token vector, one component might be highly positive and the other highly negative. If `Mx = 2λ₁c₁ + 2λ₂c₂…` where `Mv_i = λ_i v_i` and `x = Σ c_i v_i`, then even if λ_i is positive, a negative c_i can make Mx negative — which may cause the head to appear as a **negative copying head**, suppressing its own token rather than promoting it.

Despite its issues, if the eigenvectors of M tend to have large positive eigenvalues, we consider it a strong signal of a copying mechanism.

**Some very important notions:**

So far we've talked about ways of identifying types of Circuits, definitions of circuits and much more. Let us take a step back to discuss the mathematical aspects of understanding the Attention-Head Mechanism.

One main aspect that we notice in an Attention head is that, statistics or probabilities originate from primarily two regions from the abstraction of the attention head formula:

`T = W_u W_e + Σ_{h∈H} A ⊗ (W_u W_ov W_e)`

This notions that the probabilities come from two places, the Direct path, which contributes to bigram statistics (recall Week-1 work), and the second part is the Attention head mechanism!

Extending this Idea to the two-layer attention model, one obtains the following:

`T = W_u W_e + Σ_{h∈H} A ⊗ (W_u W_ov W_e) + Σ_{h1∈H} Σ_{h2∈H} (A^{h1} A^{h2}) ⊗ (W_u W_ov2 W_ov1 W_e)`

We can define the contributions as: **Direct path (Bigram)**, the **current head Attention** contribution as well as the **virtual head contribution**.

When learning about the QK circuitry, we would like to understand how things have been composed. For the sake of simplicity, we don't write out the equation. But the Scores for the Attention mechanism are generated by the contribution of 4 parts:

i. **No Composition:** The residual x portion passing through the attention mechanism via the direct path. Both the query and key come from the original residual stream, meaning no layer-1 head is involved.
ii. **Q-Composition:** The score obtained primarily by the previous layer-1 head's output flowing into the **query side**. The key still comes from the direct path, but what the token is *looking for* has been shaped by an upstream head.
iii. **K-Composition:** The score obtained primarily by the previous layer-1 head's output flowing into the **key side**. The query comes from the direct path, but how tokens *present themselves to be found* has been shaped by an upstream head. This is the dominant term in induction circuits.
iv. **Q+K-Composition:** Both the query and key sides are enriched by layer-1 heads, potentially different ones. This represents the fullest form of inter-layer interaction, where two upstream heads are jointly controlling the attention pattern of the layer-2 head.

**Some points about V-Composition:** V-composition can be thought of as passing a given attended input through two V-Matrices which end up being one Matrix (`Wx W = W'`). This plays an interesting role of residuality, forming a virtual head. This also in a sense allows for the passing of data one or two layers ago. The Virtual Heads seem quite prevalent in larger models and are responsible for complex inference tasks, for prediction.

**What is injection?**

Injection is the process of adding to the activation of a LLM, a concept vector to see whether there is change in the outcome.

**What is a concept vector? How do we obtain it?**

The best way to describe it is, a concept of a given term or group of terms is given by subtracting the activation of the terms by the mean of unrelated terms.

**What is the binary detection paradigm? What's the problem? What's the solution?**

The idea is that we ask the Large Language models whether it thinks one has injected anything, which would be a yes or no question. It is similar to Anthropic which expects it to both detect and also describe what was injected.

**What is differential sensitivity?**

Differential sensitivity is quite in contrast to binary detection, wherein a model is asked out of n (consider 10) sentences to detect some sentences whose query had been injected, they are also required to tell us the type of injection along with other things like strength etc.

Some point note that it is quite important that a model is quite in-depth and they find it different from the paper by Lindsey (2026) where Lindsey finds injection at later layers to be better, but here early layers are better.

**Some Definitions (Abridged from Lindsey (2026)):**

i. **Grounding:** The model's description of its internal state must causally depend on the aspect that is being described. That is, if the internal state were different, the description would change accordingly.
ii. **Internality:** The causal influence of the internal state on the model's description must be internal — it should not route through the model's sampled outputs. If the description the model gives of its internal state can be inferred from its prior outputs, the response does not demonstrate introspective awareness.
iii. **Meta-cognitivity:** The Model mustn't try to explain about the input or the concept that has been provided, but it has to put into words about how 'thoughts' were being generated about said emotion.

## Week 3

### Overview

The primary work this week focused on setting up the data that will be used for further analysis, as well as establishing the basic pipeline for analysis to be built upon in subsequent weeks.

The dataset utilised is a processed dataset derived from the [TruthfulQA dataset](https://huggingface.co/datasets/domenicrosati/TruthfulQA), originally designed to evaluate whether a language model can produce reliable, truthful answers in a human-like manner. The processed dataset is set up to study how the model functions and understand its internal activations for mechanistic interpretability studies, particularly in probing and identifying internal representations that distinguish truthful self-correction from hallucination.

---

### I) Rendering JSON

This step sets up the relevant data structure and file. The script loads the data from HuggingFace and packages the dataset per data unit as follows:

- **ID:** For the sake of tracking individual samples.
- **Question:** The prompt or original question posed to the LLM.
- **prompt_A:** The self-corrected (truthful) response of the LLM, provided alongside the question.
- **prompt_B:** The hallucinated (incorrect) response of the LLM, provided alongside the question.
- **source:** As per the TruthfulQA dataset.

This generates the required `contrastive_pairs.jsonl` and `contrastive_pairs.json` files.

Two important notes:
1. Only data belonging to the **Misconceptions** category has been considered.
2. A given question may have multiple hallucination prompts associated with it. The dataset is structured such that each hallucination prompt occupies its own row, paired with the appropriate question and self-corrected prompt.

---

### II) Caching

With the processed data in place, the LLMs are loaded from HuggingFace to verify data compatibility and confirm the relevant framework is set up correctly. The process is as follows:

1. Load the model from HuggingFace and set a file directory for it.
2. Load `contrastive_pairs.json` to obtain the data.
3. Pass the `prompt_A` and `prompt_B` prompts to the LLM and record activations using `run_with_cache` from TransformerLens.
4. Save the Key-Value Cache Dictionary to a `.pt` file. During this step, log time, memory usage, and other relevant details such as sub-module names for verification and future analysis.

**Note:** A `.pt` file has been used in place of a `.json` file due to file size constraints and the limitations of the JSON format for this type of data. This change is not expected to significantly affect the overall pipeline going forward.

---

### Programme Walkthrough

#### Rendering JSON

The data is loaded from the TruthfulQA dataset (via HuggingFace) and packaged into structured JSON entries as follows:

- **ID:** For the sake of tracking individual samples.
- **Question:** The prompt or original question posed to the LLM.
- **prompt_A:** The self-corrected (truthful) response of the LLM, provided alongside the question.
- **prompt_B:** The hallucinated (incorrect) response of the LLM, provided alongside the question.
- **source:** As per the TruthfulQA dataset.

This creates the `contrastive_pairs.jsonl` and `contrastive_pairs.json` files.

#### Smoke Test

As outlined in the `Onboarding.md` file, the purpose of this step is to verify that caching works correctly. The process is as follows:

1. Load the model.
2. Generate a file directory for the given model.
3. Load the two prompts — `prompt_A` and `prompt_B` — from the above JSON file(s).
4. Generate a file directory for the particular prompt ID(s).
5. Obtain the cached activations and save them as `.pt` files.

**Note:** `.pt` files have been used in place of the recommended `.json` files due to file size and JSON format constraints. Loading `.pt` files as full key-value dictionaries is considered equally effective for the overall pipeline.

#### Attached Files

- `Caching using Transformer Lens.py`
- `Rendering JSON.py`
- `contrastive_pairs.json`
- `contrastive_pairs_schema.json`


1. One thing that's well-specified and why it matters. *(to be filled)*

2. One thing that's underspecified and a concrete way to fix it. *(to be filled)*

3. One risk the protocol doesn't mention, and how you would mitigate it. *(to be filled)*

## Week 4

### An understanding of the metrics to prove that the probe accounts for introspection:
We test primarily on three aspects:
1. **Generality:** The model should perform well on data that it hasn’t seen, or rather, data that comes from a different domain.
2. **Specificity:** We would like to see whether the probe is able to detect introspection specifically, rather than generic text patterns. This is done by simply prompting with unambiguous and factual inputs and seeing whether the classifier categorizes them into a certain group. If the classifier is unsure or neutral, then we can be confident that the model actually works on detecting introspection; otherwise, the probe has learned something other than introspection, meaning that the model likely doesn’t introspect on those segments.
3. **Confound Check:** This is to see whether the probe is truly capturing internal introspective processing or whether it is merely relying on the structural template of the prompt. This is simply done by swapping the labels or reversing the underlying prompt syntax during testing to see if the classification metrics degrade significantly, thereby proving that the probe is sensitive to the true context rather than surface-level heuristics.

### Q) Why is vec(A) - vec(B) similar to that of the normed weight vector of the probe model?
The difference between the vectors essentially represents the direction from B to A. When we try to distinguish between classes or prompts using a linear probe, the model is effectively trying to capture this difference: maximizing the separation between A and B.

An interesting perspective is that the normalized weight vector of the probe corresponds to the direction perpendicular to the hyperplane that separates these classes. In other words, both the difference vector and the probe’s weight vector point along the same direction that best distinguishes the two classes in representation space.