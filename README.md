# Machine Learning
Machine Learning experiment tracking, model checkpointing.

**Document Classification Solution :**
A dual encoder / zero-shot NLI-style classifier where the labels guide semantics. This is a semantic matching model — not a classification head

Encoder architecture for the semantic dual-encoder mode.

The encoder is shared for docs and labels
The model outputs embeddings and similarity scores directly.  
LoRA adapters can be fine-tuned further via GRPO reward signals training a two-tower (dual encoder) embedding model similar to:
Zero-shot NLI (e.g., BART-NLI, DeBERTa-NLI)
The label descriptions act as hypothesis statements.

The document acts as the premise.

Semantic Matching / Bi-Encoder Architecture (Sentence-BERT style)

Where both:

•	doc text → embedding

•	label description → embeddings are produced separately, then their similarity is the prediction.

Label embeddings are semantic, not categorical
The model learns whether:

(doc_text, label_text)
semantically match — which makes the model generalize to new labels, just like NLI-based zero-shot classifiers.

**1. Shared Encoder Architecture:**

o	One AutoModel encoder is used for both documents and label descriptions.

o	LoRA adapters are applied for efficient fine-tuning without modifying the base model.

**2.	Label Description Embeddings:**

o	Label descriptions are tokenized once and embedded via the same encoder.

o	During inference, document embeddings are compared to label embeddings using cosine similarity.

**3.	Zero-Shot Classification Style:**

o	Cosine similarity between document and label embeddings is scaled (COSINE_SCALE) and passed through sigmoid.

o	BCEWithLogitsLoss allows multi-label supervision, but the structure remains zero-shot at inference because the model computes similarity to label descriptions rather than classifying into fixed IDs.

**4.	Inference Functions:**

o	predict_labels embeds a new document and compares it to pre-computed label embeddings.
o	Multi-label zero-shot predictions are generated purely based on semantic similarity, consistent with dual-encoder zero-shot classifiers.
o	In short: LoRA + PBT updates the encoder efficiently, but the dual-encoder zero-shot classification logic is fully preserved.

**Components:**

1. New label description dictionary (already created)

Semantic-rich descriptions.

2. A dual-encoder dataset class
•	Encodes documents
•	Encodes label descriptions once
•	No classifier head
•	Uses cosine similarity predictions
•	Uses BCE + contrastive semantic loss

3. A new model wrapper
•	Loads bge-m3
•	Adds LoRA
•	Output = embeddings
•	Computes cosine similarity internally

4. A new training loop
•	Ray Tune PBT-compatible
•	Uses existing trainer
•	But updated to support dual encoder logic

5. Inference API
•	Embed document
•	Compare to fixed label embeddings
•	Output similarities + thresholds

6. Feedback/Reward Model : A GRPO-ready interface
•	get_doc_embedding()
•	get_label_embeddings()
•	cosine similarity reward
•	RL loop-ready structure

Everything will be designed explicitly so that GRPO can continue optimizing LoRA modules on top of the trained semantic encoder.

In nutshell this is what been done

•	The labels are represented by their full semantic descriptions embedded once.

•	Documents are embedded and similarity is computed by cosine similarity between embeddings.

•	Loss is BCEWithLogits on similarity scores vs multi-hot labels.

•	LoRA is applied to the shared transformer encoder.

•	PBT tunes hyperparameters with Ray Tune.

•	Model is trained end-to-end to maximize semantic alignment between docs and labels.

•	Prediction is done by thresholding cosine similarity scores after sigmoid.

**Environment : Databricks Cluser**

Worker Type: Standard_NV36adms_A10_v5
Each Worker has 1x NVIDIA A10 GPU with 880 GB RAM
This is a multi-node GPU cluster (not a single machine with 4 GPUs)
Databricks automatically provisions 4 GPU nodes
Total GPUs available = 4 nodes × 1 GPU each = 4 GPUs
This is a multi-node GPU cluster (not a single machine with 4 GPUs)

**Training Architecture**

<img width="975" height="613" alt="image" src="https://github.com/user-attachments/assets/d9094a9a-628c-4c0d-ab4f-329424de48d6" />

 **Extended Training Architecture**
 
 <img width="975" height="630" alt="image" src="https://github.com/user-attachments/assets/bd22b37a-1ff5-4dd4-acdb-1de543e2aa9b" />

**With Feedback and Policy GRPO Training**

•	Builds on Dual-Encoder + LoRA + Cosine BCE + Ray-PBT diagram.

•	Adds the GRPO stage:
o	SFT + LoRA best checkpoint saved as a Reference Model

o	Cloned into a Policy Model

o	GRPOTrainer uses:
Bernoulli log-probs for multi-label outputs

Feedback scores (1–5) as rewards from a feedback store
Produces an Updated Policy Model (SFT + GRPO LoRA weights)

<img width="975" height="591" alt="image" src="https://github.com/user-attachments/assets/793d68c9-e945-493a-bc50-94bee406c56a" />

Model Lifecycle for the above Training Architecture

<img width="975" height="682" alt="image" src="https://github.com/user-attachments/assets/ba236f54-0444-4b05-bb99-29ba8a468cd2" />

**The above Model Lifecycle shows how models evolve in this pipeline:**

1.	Pretrained BGE encoder
2.	SFT + LoRA training with PBT
3.	Deploy SFT+LoRA classifier (v1)
4.	Collect user feedback during usage
5.	Offline GRPO training (ref vs policy, feedback-based rewards)
6.	Evaluate candidate SFT+GRPO policy
7.	Deploy v2 policy, archive old checkpoints, repeat loop.


•	Builds on top of Dual-Encoder + LoRA + Cosine BCE + Ray-PBT diagram.
•	Adds the GRPO stage:
o	SFT + LoRA best checkpoint saved as a Reference Model
o	Cloned into a Policy Model
o	GRPOTrainer uses:
   Bernoulli log-probs for multi-label outputs
   Feedback scores (1–5) as rewards from a feedback store
   Produces an Updated Policy Model (SFT + GRPO LoRA weights)
Model Lifecycle Diagram
•	Shows how models evolve in this pipeline:

1.	Pretrained BGE encoder
2.	SFT + LoRA training with PBT
3.	Deploy SFT+LoRA classifier (v1)
4.	Collect user feedback during usage
5.	Offline GRPO training (ref vs policy, feedback-based rewards)
6.	Evaluate candidate SFT+GRPO policy
7.	Deploy v2 policy,
8.	Archive old checkpoints, repeat loop.

-----------------------------------------------------------------------------------------------------------------------------------------------------
**Training 1 – Binary Cross-Entropy+Contrastive Loss as Loss Function**

The training process in this code is designed to fine-tune a pre-trained language model (specifically, a dual-encoder model) to classify documents as either tax problem or tax solutions.

The model is trained on a dataset of labeled documents, where each document is associated with a label indicating whether it's a tax question or a tax solution or similar tax related markers which help identify a tax document.

The model learns to represent each document as a dense vector (embedding) and then uses these embeddings to predict the label.

The effectiveness of this training process depends on several factors, including:

**1.	Quality of the dataset:** The dataset should be large, diverse, and well-labeled, with a good balance of tax questions and tax solutions.

**2.	Choice of pre-trained model:** The pre-trained model should be suitable for the task at hand, and the dual-encoder architecture is a good choice for this type of classification task.

**3.	Hyperparameter tuning:** The hyperparameters, such as learning rate, batch size, and number of epochs, should be carefully tuned to optimize the model's performance.

**4.	Evaluation metrics:** The model's performance should be evaluated using relevant metrics, such as accuracy, precision, recall.

Code used in this training process is well-designed and the model is being fine-tuned using a suitable pre-trained model and a reasonable set of hyperparameters.

However, to determine whether this training process is effective for this specific use case, we'll need to evaluate the model's performance on a held-out test set and consider the following factors:

**1.	Accuracy:** How accurate is the model in classifying documents as tax questions or tax solutions?

**2.	Precision:** How precise is the model in identifying true positives (i.e., documents that are actually tax questions or tax solutions)?

**3.	Recall:** How well does the model recall true positives (i.e., documents that are actually tax questions or tax solutions)?

**4.	F1-score:** What is the F1-score, which balances precision and recall?

If the model's performance is satisfactory, we can use it to classify new unseen documents as tax questions or tax solutions. However if the performance is not satisfactory, we may need to adjust the training process, such as by tweaking the hyperparameters, using a different pre-trained model, or collecting more data.

**Code updates the hyperparameters using Population-Based Training (PBT) scheduler from the Ray Tune library.**

In this code the PBT scheduler is defined as follows:

<img width="658" height="492" alt="image" src="https://github.com/user-attachments/assets/ba4beaab-b63e-43e6-8a5f-f8b7a6f87fdd" />

 

This scheduler will perturb the hyperparameters every 6 training iterations and the perturbations are defined as follows:

•	lr: log uniform distribution between 1e-6 and 5e-5

•	wd: uniform distribution between 0 and 0.1

•	warmup: uniform distribution between 0 and 0.2

•	lora_rank: one of the values [4, 8, 16, 32]

The tune.run function is then used to run the training function train_with_pbt with the PBT scheduler:

<img width="802" height="611" alt="image" src="https://github.com/user-attachments/assets/083083c9-5cb5-49ab-bdc9-592a0692828b" />

This will run 4 trials of the training function with the initial hyperparameters and then the PBT scheduler will perturb the hyperparameters and run new trials with the updated hyperparameters.


The best trial with the minimum loss is then selected and the corresponding model is saved:
 
<img width="975" height="79" alt="image" src="https://github.com/user-attachments/assets/06857906-0878-4066-8324-35d161e4d5a3" />


**How is Loss calculated in this Training?**

The loss is calculated in the compute_loss method of the SemanticDualEncoderTrainer class.

<img width="975" height="835" alt="image" src="https://github.com/user-attachments/assets/12dfd117-57c3-4bcc-9ffa-1e3cbf02465f" />


**The loss is calculated as a combination of two components:**

**1.	Binary Cross-Entropy (BCE) Loss:** This is calculated using the nn.BCEWithLogitsLoss() function, which takes the scaled cosine similarity (cos_scaled) and the target values (targets) as inputs.
   
**2.	Contrastive Loss:** This is calculated using the formula (pos * (1 - cos).clamp(min=0) + neg * (cos - margin).clamp(min=0)).mean(), where pos and neg are the positive and negative target values respectively, and margin is a hyperparameter set to 0.2.

The final loss is the sum of the BCE loss and the contrastive loss weighted(multiplied) by the contrastive_weight hyperparameters(0.05,0.1,0.2,0.5,1.0).

The cos variable represents the cosine similarity between the document and label embeddings and the emb variable represents the document embeddings.

These values are returned along with the loss if return_outputs is True.

**What is Binary Cross-Entropy Loss?**

Binary Cross-Entropy (BCE) loss is a common loss function used in binary classification problems, where the target variable is a binary label (0 or 1). It measures the difference between the predicted probabilities and the true labels.

**Mathematical Formula**

The BCE loss is calculated as follows:

L(y, y_pred) = -[y * log(y_pred) + (1-y) * log(1-y_pred)]
where:
•	y is the true label (0 or 1)
•	y_pred is the predicted probability of the positive class (between 0 and 1)
•	log is the natural logarithm


**Example**
Suppose we have a binary classification problem, where we want to predict whether a person is likely to buy a product (label 1) or not (label 0). We have a model that outputs a probability of buying the product, which we'll call y_pred.

Let's say we have two examples:

True Label (y)	Predicted Probability (y_pred)
1	0.8
0	0.3

To calculate the BCE loss for each example, we plug in the values:

**Example 1: True Label = 1, Predicted Probability = 0.8**
L(1, 0.8) = -[1 * log(0.8) + (1-1) * log(1-0.8)] L(1, 0.8) = -[1 * log(0.8) + 0 * log(0.2)] L(1, 0.8) = -log(0.8) L(1, 0.8) ≈ 0.223

**Example 2: True Label = 0, Predicted Probability = 0.3**
L(0, 0.3) = -[0 * log(0.3) + (1-0) * log(1-0.3)] L(0, 0.3) = -[0 * log(0.3) + 1 * log(0.7)] L(0, 0.3) = -log(0.7) L(0, 0.3) ≈ 0.356.

The BCE loss for each example is approximately 0.223 and 0.356, respectively. The goal of the model is to minimize the BCE loss, which means it should try to predict probabilities that are close to the true labels.

In this example, the model is doing a decent job, but there's still room for improvement. If the model were to predict y_pred = 1.0 for the first example and y_pred = 0.0 for the second example, the BCE loss would be 0 for both examples, which is the minimum possible value.

The nn.BCEWithLogitsLoss() function is used, which is a combination of a sigmoid activation function and the binary cross-entropy loss function.

The nn.BCEWithLogitsLoss() function takes the logits (i.e., the output of the model before applying the sigmoid function) as input, applies the sigmoid function internally, and then calculates the binary cross-entropy loss.

So, while the sigmoid function is not explicitly used in this code, it is implicitly applied by the nn.BCEWithLogitsLoss() function.

**Here's a breakdown of what's happening:**

The model outputs a value, which is often referred to as the "logit".

The nn.BCEWithLogitsLoss() function takes this logit as input. Internally, the nn.BCEWithLogitsLoss() function applies the sigmoid function to the logit, which maps the logit to a probability between 0 and 1.

The binary cross-entropy loss is then calculated between the predicted probability and the true label.

By using nn.BCEWithLogitsLoss(), the code is effectively using the **sigmoid function as the activation function**, but it's done internally by the loss function, rather than explicitly applying the sigmoid function to the model output.

In the compute_loss method of the SemanticDualEncoderTrainer class, the line bce = nn.BCEWithLogitsLoss()(cos_scaled, targets) is where the sigmoid function is implicitly applied.The cos_scaled variable is the logit, and the nn.BCEWithLogitsLoss() function applies the sigmoid function to it, and then calculates the binary cross-entropy loss between the resulting probability and the targets variable.


**What is Contrastive Loss and how is it calculated?**

Contrastive loss is used as a regularization term to encourage the model to produce embeddings that are close together for similar inputs (e.g., documents and labels that are related) and far apart for dissimilar inputs.

**The contrastive loss is calculated as follows:**

<img width="550" height="281" alt="image" src="https://github.com/user-attachments/assets/76edd87b-5c48-4a2a-a859-a4714842748e" />

Here is a breakdown of the components:

•	**margin**: a hyperparameter that controls the minimum distance between dissimilar embeddings. In this case, it's set to 0.2.

•	**pos**: the positive targets (i.e., the labels that indicate a relationship between the document and label).

•	**neg**: the negative targets (i.e., the labels that indicate no relationship between the document and label).

•	**cos**: the cosine similarity between the document and label embeddings.

•	**contrast**: the contrastive loss term.

The contrastive loss is calculated as the sum of two terms:

**1.	pos * (1 - cos).clamp(min=0):** This term encourages the model to produce embeddings that are close together for similar inputs. When the cosine similarity is high (i.e., the embeddings are similar), this term is small. When the cosine similarity is low (i.e., the embeddings are dissimilar), this term is large.
   
**2.	neg * (cos - margin).clamp(min=0):** This term encourages the model to produce embeddings that are far apart for dissimilar inputs. When the cosine similarity is low (i.e., the embeddings are dissimilar), this term is small. When the cosine similarity is high (i.e., the embeddings are similar), this term is large, but only if the cosine similarity is greater than the margin.The clamp(min=0) function ensures that the loss terms are non-negative.

**Example**

Suppose we have two document-label pairs:
•	Document 1: "This is a great product"
•	Label 1: "Product Review"
•	Document 2: "I love this restaurant"
•	Label 2: "Restaurant Review"

The model produces the following embeddings:
•	Document 1 embedding: [0.7, 0.3, 0.1]
•	Label 1 embedding: [0.8, 0.2, 0.1]
•	Document 2 embedding: [0.2, 0.6, 0.3]
•	Label 2 embedding: [0.1, 0.7, 0.4]

The cosine similarities between the embeddings are:
•	Document 1 and Label 1: 0.9
•	Document 2 and Label 2: 0.8
•	Document 1 and Label 2: 0.2
•	Document 2 and Label 1: 0.1

The contrastive loss would be calculated as follows:
•	pos * (1 - cos).clamp(min=0):
o	Document 1 and Label 1: 0.1 * (1 - 0.9) = 0.01
o	Document 2 and Label 2: 0.1 * (1 - 0.8) = 0.02
•	neg * (cos - margin).clamp(min=0):
o	Document 1 and Label 2: 0.9 * (0.2 - 0.2) = 0
o	Document 2 and Label 1: 0.9 * (0.1 - 0.2) = 0

The contrastive loss would be the sum of these terms: 0.01 + 0.02 + 0 + 0 = 0.03.

The model would be encouraged to produce embeddings that are closer together for similar inputs (e.g., Document 1 and Label 1) and farther apart for dissimilar inputs (e.g., Document 1 and Label 2).

**Activation Functions:**

**tanh and Sigmoid used in this training - Practical explanation on where it is used within the code.**

The **pooler_output** uses a tanh activation function and the BCEWithLogitsLoss() function uses a **sigmoid activation** function internally.

However these two activation functions are not in conflict with each other. Here is why:

**Pooler output:** The pooler_output is used to extract a fixed-size vector representation of the input sequence. The tanh activation function is applied to the last hidden state of the encoder to produce this vector. This is a common technique used in many transformer-based models.

**BCEWithLogitsLoss():** The BCEWithLogitsLoss() function is used to compute the binary cross-entropy loss between the model's output and the target labels. This function uses a sigmoid activation function internally to produce a probability output.

**The key point to note is that these two activation functions are applied at different stages of the model:**

The **tanh activation function** is applied to the pooler_output to produce a vector representation of the input sequence.

The **sigmoid activation function** is applied to the output of the model (i.e., the logits) to produce a probability output which is then used to compute the binary cross-entropy loss.

Since these two activation functions are applied at different stages, **they do not conflict with each other.** In fact, this is a common pattern in many deep learning models, where different activation functions are used at different stages to achieve specific goals.

**To illustrate this, consider the following sequence of operations:**


**Input sequence → Encoder → Last hidden state → Tanh activation → Pooler output (vector representation)**

Pooler output → Linear layer → Logits → Sigmoid activation (internal to BCEWithLogitsLoss()) → Probability output → Binary cross-entropy loss

As we can see, the tanh activation function is applied to the pooler_output to produce a vector representation while the sigmoid activation function is applied to the logits to produce a probability output. These two activation functions are not in conflict with each other and they serve different purposes in the model.

**The two plots which are empty why?**

Mutating lora_rank during PBT is fundamentally different from mutating continuous hyperparameters like lr or margin and here's why it destabilizes training:

Rank is architectural not a training knob. LoRA decomposes weight updates as ΔW = B·A where B is (d × r) and A is (r × d). Changing rank from 32 → 8 means the matrices have incompatible shapes. When PBT tries to exploit (copy weights from a top-performing trial to a bottom trial), it can't transfer LoRA weights between trials with different ranks — the tensors don't match.

Even without exploitation, mutation is destructive. If PBT perturbs rank mid-training, the LoRA adapter must be reinitialized with new dimensions. This instantly erases all learned low-rank adaptations accumulated over previous iterations — effectively resetting that trial's model to near-random LoRA state while the rest of training expects a warm-started model.

Contrast with safe-to-mutate hyperparameters:

lr, wd, warmup — optimizer state adjusts smoothly; no weight destruction
alpha, margin, contrast_weight — loss scaling changes; gradients shift but model weights remain intact
The fix of having lora_rank as 32 is the standard practice: hardcode r=32 and lora_alpha=32 so all trials share identical architecture, making PBT exploitation (weight copying between trials) safe. So #"lora_rank": [4,8,16,32] would have caused checkpoint-incompatible trials and likely RuntimeError: size mismatch or sudden loss spikes after exploitation events.

Following the above training with Trials running in parallel : 

<img width="789" height="257" alt="image" src="https://github.com/user-attachments/assets/98b897f1-cd3e-46ae-887f-20a141f14f52" />

**Training metrics - Loss improvement per trial for all hyperparameters**

<img width="1330" height="420" alt="image" src="https://github.com/user-attachments/assets/ba4baf0e-8753-481a-8bac-6ccf2b50a8b9" />


**Parameters Perturbating:**

<img width="1298" height="71" alt="image" src="https://github.com/user-attachments/assets/2dbdeb5e-86af-4922-83ee-67d2afe3c46b" />

Plots:

<img width="1354" height="899" alt="image" src="https://github.com/user-attachments/assets/7696410d-72de-4d2f-bfe6-f8f4b3a00d7a" />

As we can see from the above Population-Based Training (PBT) works by periodically perturbing hyperparameters based on performance. The plots confirm this:

1) lr (learning rate): Starts constant, then jumps significantly after a few iterations.
2) wd (weight decay): Shows a clear change mid-training.
3) warmup, margin, alpha: Also exhibit sudden changes at certain iterations.
4) lora_rank: Changes from 4 → 32 or vice versa in some trials.

These jumps indicate PBT exploitation and exploration steps, where poorly performing trials copy weights from better ones and perturb hyperparameters slightly. And Perturbation helps escape **local minima** and adapt hyperparameters dynamically. The fact that we see sharp changes means PBT is actively tuning parameters during training.

Why hyperparameters changed the most?
**From the extracted ranges (normalized 0–1 scale):**
Learning Rate (lr): Range ≈ 0.90 → most perturbed

Warmup: Range ≈ 0.90 → highly perturbed

Margin: Range ≈ 0.90 → highly perturbed

Weight Decay (wd): Range ≈ 0.64 → moderate

LoRA Rank: Range ≈ 0.58 → moderate

Alpha: Range ≈ 0.16 → barely changed

The orange trial barely moved (range ≈ 0.003 for all parameters), confirming that only the blue trial underwent significant perturbations.

**Example PBT Hyperparameter Evolution (actual values) per trial:**

<img width="1323" height="666" alt="image" src="https://github.com/user-attachments/assets/a8f02657-83bf-4fe1-b0b5-8eb4b6da19ba" />


<img width="1018" height="715" alt="image" src="https://github.com/user-attachments/assets/d5112a04-26fb-4557-a963-d759be20026b" />

<img width="1124" height="630" alt="image" src="https://github.com/user-attachments/assets/bdfc00b5-fae4-44bd-a6c5-9523667a8d6a" />

<img width="1112" height="427" alt="image" src="https://github.com/user-attachments/assets/e4b8616d-bfac-4640-8083-c205e71c8a05" />


-----------------------------------------------------------------------------------------------------------------------------------------------------

**Ray Tune experiment analysis for hyperparameter tunning.**

**Best trial identification - Which trial achieved the lowest loss and what were its final hyperparameters?**


<img width="1332" height="868" alt="image" src="https://github.com/user-attachments/assets/7262ea28-33a6-429c-b288-1cc9968fbea5" />


<img width="1325" height="773" alt="image" src="https://github.com/user-attachments/assets/4bd166d3-9ba5-4ecd-9bc5-a401ce41d8d5" />


**Convergence analysis - How quickly did the trials converge?**

<img width="1303" height="422" alt="image" src="https://github.com/user-attachments/assets/b261e2f7-9159-4287-8470-651109d184e6" />

<img width="1332" height="495" alt="image" src="https://github.com/user-attachments/assets/496fd8d6-3ba9-4703-8928-9e9fb675ffd7" />

<img width="1331" height="666" alt="image" src="https://github.com/user-attachments/assets/f3413e2a-3f49-4c44-8021-8b8fcd7ec89d" />



-----------------------------------------------------------------------------------------------------------------------------------------------------
**Training 2 : Implementing a dual-encoder model with Low-Rank Adaptation (LoRA) and Population-Based Training (PBT) using the Ray Tune library.**

Loss is calculated in the SemanticDualEncoderTrainer class, specifically in the compute_loss method.

<img width="975" height="1109" alt="image" src="https://github.com/user-attachments/assets/18482020-1369-4732-8018-4e571d56c43c" />


**The loss is calculated as follows:**

1.	The model outputs are passed through the compute_loss method, which extracts the target and soft labels from the inputs.
2.	The model outputs are then passed through the model to get the embeddings for the documents and labels.
3.	The embeddings are L2-normalized to have a length of 1.
4.	The cosine similarity between the document and label embeddings is calculated.
5.	The cosine similarity is scaled to logits using a temperature parameter.
6.	The BCE loss is calculated between the logits and the hard labels (targets) and soft labels (soft).
7.	The final loss is calculated as the average of the hard loss and soft loss.
The loss is a combination of the BCE loss between the logits and the hard labels, and the BCE loss between the logits and the soft labels. The soft labels are used to provide additional information to the model, and the hard labels are used to provide a clear target for the model to learn.

In the context of the training, hard labels and soft labels are used to train the model.

**Hard Labels:** Hard labels are binary labels (0 or 1) that indicate whether a document is relevant or not relevant to a given label. 

For example, if we're training a model to classify documents as "relevant" or "not relevant" to a particular topic, the hard labels would be:
•	1: Relevant
•	0: Not Relevant

**Soft Labels:** Soft labels on the other hand are continuous values between 0 and 1 that represent the degree of relevance or similarity between a document and a label. Soft labels are used to provide more nuanced information about the relationship between the document and the label.

For example, if we're training a model to classify documents as "relevant" or "not relevant" to a particular topic the soft labels could be:

•	0.8: Highly relevant
•	0.4: Somewhat relevant
•	0.2: Not very relevant
•	0.0: Not relevant at all

Example: Let's say when training a model to classify documents as "relevant" or "not relevant" to the topic of "machine learning". Need to have a document that mentions "deep learning" and "neural networks", but doesn't explicitly mention "machine learning".

•	Hard label: 1 (Relevant) or 0 (Not Relevant)

•	Soft label: 0.6 (Somewhat relevant, since the document mentions related topics, but not explicitly "machine learning")

In this example, the hard label would be either 1 or 0, indicating whether the document is relevant or not. The soft label, on the other hand, would be 0.6, indicating that the document is somewhat relevant to the topic of machine learning, but not explicitly.

**Why use both hard and soft labels?**
Using both hard and soft labels can help the model learn more nuanced relationships between documents and labels. The hard labels provide a clear indication of whether a document is relevant or not while the soft labels provide more detailed information about the degree of relevance.

In the context of the training code, the target variable represents the hard label and the soft variable represents the soft label. The model is trained to predict both the hard label and the soft label using the BCEWithLogitsLoss function to compute the loss.

Here's an example of how the labels might be used in the training code:
 
In this example the targets variable represents the hard label and the soft variable represents the soft label. The model is trained to predict both the hard label and the soft label using the BCEWithLogitsLoss function to compute the loss. The hard and soft losses are combined using a weighted average with a weight of 0.5 for each.

The choice of loss function depends on the specific problem trying to solve and the characteristics of the training data. Here's a brief analysis of the options mentioned:

**1.	Binary Cross-Entropy (BCE) Loss:** BCE loss is a common choice for binary classification problems where the goal is to predict one of two classes (e.g., tax problem or not). BCE loss measures the difference between the predicted probabilities and the true labels. It's a good choice when:

o	The classes are mutually exclusive (i.e., a document can't be both a tax problem and a solution).

o	The classes are balanced (i.e., roughly equal number of positive and negative examples).

**2.	Contrastive Loss:** Contrastive loss is a type of loss function that encourages the model to learn embeddings that are close together for similar examples (e.g., documents with similar tax problems) and far apart for dissimilar examples (e.g., documents with different tax problems). Contrastive loss is a good choice when:

o	Model needs to learn a representation of the data that captures the underlying structure (e.g., tax problems and solutions).
o	There is a large number of classes or a complex classification problem.

**3.	Dual-Encoder Model with Cosine Similarity and BCE Loss:** The approach implemented uses a dual-encoder model to learn two separate embeddings for documents and labels. The cosine similarity between these embeddings is used to compute the loss which is then combined with BCE loss. This approach is a good choice when:

o	The Model needs to learn a representation of the data that captures the similarity between documents and labels.
o	There is  a large number of labels or a complex classification problem.

Considering specific problem recommend using a combination of BCE loss and contrastive loss. Here's why:

•	BCE loss can help the model learn to distinguish between tax problems and solutions, which is a binary classification problem.

•	Contrastive loss can help the model learn a representation of the data that captures the underlying structure of tax problems and solutions which can improve the overall performance of the model.

-----------------------------------------------------------------------------------------------------------------------------------------------------

**Training and Supervised FineTuning for a Classification Problem - Calculating the f1_macro score**

In a Supervised FineTunning model specifically in a Classification problem the F1-macro is an evaluation metric it is often monitored during supervised fine-tuning (SFT) to measure how well the encoder model is learning to classify. The F1 score is the harmonic mean of precision and recall for a class. When fine tuning a model the training objective is cross-entropy loss specifically in this case where we have multiple independent labels like problem, solution, tax type, tax topic and tax year the correct one is Binary Cross-Entropy(BCE) also can be called as Sigmoid + BCE loss which is the standard for multi-label classificaiton problem and this is from where the gradient is computed and F1_macro metric is computed after each epoch (or batch) as a validation metric not as a loss like in RL where a reward signal directly drives optimization (e.g. in RLHF or GRPO), F1-macro is only used for monitoring and model selection - it does not produce gradients. It tells if the model is improving across all classes fairly.

| Stage                       | Metric used                           |
| --------------------------- | ------------------------------------- |
| **Training / Back-propagation**     | Cross-Entropy (for classification)    |
| **Validation / Evaluation** | F1-macro, Accuracy, Precision and Recall |

In short

| Type                            | Example                                                                                                       | Loss Function                      |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| **Single-label classification** | Each document has *only one* label (e.g., “Tax Type = Corporate”)                                             | **Softmax + Cross-Entropy**        |
| **Multi-label classification**  | Each document can have *multiple* labels (e.g., “has_problem=1, has_solution=1, tax_topic=‘TransferPricing’”) | **Sigmoid + Binary Cross-Entropy** |

**How BCE works for tax classifier**

For each label (output neuron), we compute:

<img width="983" height="226" alt="image" src="https://github.com/user-attachments/assets/468f6aca-957b-4794-8a20-cb909d27183e" />

Each label has its own independent sigmoid, so the model can output:

| Label                       | Meaning         | Example Output |
| --------------------------- | --------------- | -------------- |
| `has_problem`               | 0/1             | 0.89           |
| `has_solution`              | 0/1             | 0.65           |
| `tax_type_Corporate`        | one-hot (multi) | 0.80           |
| `tax_topic_TransferPricing` | 0/1             | 0.74           |
| `tax_year_2023`             | 0/1             | 0.55           |

As when compare softmax with CrossEntropy for  a single-label scenario

| Component          | Description                   | Softmax                     | Sigmoid                           |
| ------------------ | ----------------------------- | --------------------------- | ----------------------------------- |
| **Loss**           | Training objective            | CrossEntropy (single-label) | **BCEWithLogitsLoss (multi-label)** |
| **Activation**     | Output layer                  | Softmax                     | **Sigmoid (per label)**             |
| **Metric**         | Eval metric                   | argmax → F1                 | **sigmoid + threshold → F1_macro**  |
| **Regularization** | Weight decay + early stopping | OK                        | OK                                |


Example : Suppose a document both describes a tax problem and provides a solution. Softmax forces probabilities to sum = 1.0 -> the model must choose only one label (here “tax_problem”) even though “tax_solution” is also correct as shown in the table below where Softmax forces the probabilities to sum 1.0 meaning the model must choose only one label bere which is tax_problem even though tax_solution is also correct.

| Label        | True | Model (softmax probs) |
| ------------ | ---- | --------------------- |
| tax_problem  |  1  | 0.55                  |
| tax_solution | 1  | 0.40                  |
| tax_type     | 0    | 0.03                  |
| tax_topic    | 0    | 0.01                  |
| year         | 0    | 0.01                  |

In the case of Multi-Label where a document can contain a tax problem or solution or other class we will need Sigmoid with BinaryCrossEntropy to allow multiple labels to be "on" simultaneously so that Sigmoid gives independent probabilities in this case multiple outputs can be 1 simultaneously.

| Label        | True | Model (sigmoid probs) | Pred (> 0.5) |
| ------------ | ---- | --------------------- | ------------ |
| tax_problem  | 1  | 0.85                  | 1          |
| tax_solution | 1  | 0.74                  | 1          |
| tax_type     | 0    | 0.10                  | 0            |
| tax_topic    | 0    | 0.05                  | 0            |
| year         | 0    | 0.08                  | 0            |

**Why Sigmoid with BinaryCrossEntropy classification is robust**

| Property           | Softmax + Cross-Entropy  | Sigmoid + BCE                        |
| ------------------ | ------------------------ | ------------------------------------ |
| Mutual exclusivity | Forces exactly one class | Allows any combination               |
| Probabilities      | Sum = 1                  | Each label independent (0–1)         |
| Loss               | CrossEntropyLoss         | BCEWithLogitsLoss (includes sigmoid) |
| Appropriate for    | One label per sample     | Multiple possible labels per sample  |

With BCE each of five output neurons behaves like a binary detector for its label.

Sigmoid + BCE setup is ideal with each output neuron makes an independent yes/no decision allowing multiple labels to be active for the same document.

Visuallization of how multi-label document classifier works internally

<img width="596" height="553" alt="image" src="https://github.com/user-attachments/assets/bd5edd07-8fe5-4cae-ae75-fe2f298fd549" />

A realistic neural network diagram with output neuron applies Sigmoid activation producing independent probabilities (e.g., 0.87, 0.90, 0.73, 0.12, 0.58) for each of the labels and these probabilities are compared against ground truth labels using Binary Cross-Entropy (BCE) summed over all outputs.

<img width="710" height="700" alt="image" src="https://github.com/user-attachments/assets/55980f89-2a1b-44ab-ae55-2a9a73293e3e" />

With a more detailed explanation

<img width="1255" height="806" alt="image" src="https://github.com/user-attachments/assets/ab32b2c7-a6ee-440a-a276-99a226da56db" />

**F1-macro evaluation metric code**

<img width="596" height="263" alt="image" src="https://github.com/user-attachments/assets/a3cc9ed1-ad94-4116-b816-daee9a426291" />

**Training with F1 macro for evaluation + utilize the early stopping as regularization technique**

<img width="990" height="853" alt="image" src="https://github.com/user-attachments/assets/60a33364-5319-4a7d-a544-7b263627f04f" />

<img width="1022" height="151" alt="image" src="https://github.com/user-attachments/assets/7185fec8-e121-4327-a46d-09405add3e6e" />

**Make a small validation set (e.g., 10–20%):**

dataset = Dataset.from_list(train_data).train_test_split(test_size=0.15, seed=42)

tokenized_train = dataset["train"].map(preprocess_function, batched=True)

tokenized_val = dataset["test"].map(preprocess_function, batched=True)

**Targets to look for**

If val loss keeps dropping and F1_macro climbs past ~0.70+, keep training.
If val loss stops improving for ~3 evals, stop (early stopping will do it).
If training loss ↓ but val loss ↑, we are overfitting → reduce epochs or lower LR (e.g., 1e-5) and add weight_decay.
If progress stalls Try more data per class (class balance matters).
Slightly lower LR (1e-5) or increase warmup_ratio.
Increase batch size if GPU permits (stabilizes training).
Check text length—keep max_length=256 unless the chunks are longer.

**Note: F1_macro is the macro-averaged F1 score. It is the unweighted mean of the F1 scores computed independently for each class(Problem, Solution, Topic, Tax Year) in a multi-class classification problem.**

**Why is it important?** 
It treats all classes equally, regardless of their frequency in the dataset.

It is especially useful when we have class imbalance, as it does not let dominant classes overshadow minority classes.

It provides a single metric that reflects the model’s ability to correctly classify all classes, not just the most common ones.

In our code: We are using f1_macro as the metric for early stopping and model selection, ensuring the model performs well across all tax-related categories, not just the majority class.

Current loss (1.11) shows learning, but it’s not “done.”

Add validation + metrics, keep training until val loss/metrics converge.

Expect clear gains with another 1–3 epochs and proper early stopping.

**Another Early Stopping Technique :** 

The loss is considered to not have improved if the current loss is not less than the previous loss minus a small delta (early_stop_min_delta). This allows for some minor fluctuations in the loss without triggering early stopping.

In other words, the early stopping condition is triggered when the loss has not decreased by at least early_stop_min_delta for early_stop_patience consecutive iterations.
So, to answer the question, the early stopping kicks in when the loss did not improve by at least early_stop_min_delta for more than three iterations.

Here early_stop_min_delta is set to 0.001.
So the condition is:
The loss at the current iteration should be greater than or equal to the loss at the previous iteration minus 0.001.

This condition should be met for 3 consecutive iterations.

Only then will the early stopping be triggered.

This is a common technique to prevent early stopping from being triggered by minor fluctuations in the loss.

**After further training**

<img width="478" height="162" alt="image" src="https://github.com/user-attachments/assets/a67c8b66-2639-4ee4-92a5-4189860c7c2e" />

**Core conceptual difference between how embedding similarity models (like original intfloat/multilingual-e5-base) and fine-tuned classification models (like finetuned_model_inference) work.**

Embedding + Cosine Similarity Logic (What we had)

** Goal: measure semantic closeness between any two pieces of text, without explicit labels.**

Mechanism

The model (e.g., intfloat/multilingual-e5-base) converts text into a high-dimensional vector → an embedding (e.g., 768-D float vector).


"tax problem about dividends" → [0.11, -0.02, 0.33, ..., 0.87]

We store or compute embeddings for reference prompts (“problem”, “solution”, etc.).

For a new text, compute its embedding, then take cosine similarity with each reference:
sim = dot(a,b) / (||a|| * ||b||)

Values range from -1 (opposite meaning) to 1 (same meaning).

The class with the highest similarity above a threshold (e.g., 0.8) is chosen.

**Example**

Label	Reference Text	Cosine Similarity

problem	"This describes a tax problem"	0.91

solution	"This gives a solution"	0.60

topic	"This discusses participation exemption"	0.55

year	"This refers to tax year"	0.12


→ Classified as Tax Problem
**Characteristics**
Works without supervision — no need for labeled data.

We can compare any text to any other text (universal).

But classification is approximate; it relies on semantic proximity, not learned decision boundaries.

Sensitive to the prompt wording of reference texts.

**2️) Fine-Tuned Classifier Logic (What we have now after supervised fine tuning on labelled dataset)**


**Goal:** predict explicit class probabilities learned from labeled examples (problem, solution, topic, year).

**
Mechanism
**

Start from a pretrained model (like E5) and add a classification head (a small linear layer mapping embeddings → logits for 4 classes).

Fine-tune on labeled pairs:

"This describes a tax problem …" → label=problem

"This provides a tax solution …" → label=solution

After training, the model directly outputs class logits — one scalar per label:


logits = [-0.8, 1.3, 0.2, -0.7]


Apply softmax to convert logits → probabilities:


probs = [0.10, 0.68, 0.16, 0.06]

The class with the highest probability is the predicted label.

**Example**

Label	Probability

problem	0.10

solution	0.68

topic	0.16

year	0.06

→ Classified as Tax Solution

**Characteristics**

Supervised — learns from labeled examples.

Directly optimized to minimize misclassification.

Learns nonlinear decision boundaries between classes.

Doesn’t compute vector similarity — it outputs class scores.

**How They Differ Mathematically**
| Concept          | Embedding Similarity                 | Fine-Tuned Classifier         |
| ---------------- | ------------------------------------ | ----------------------------- |
| Output           | Vector (e.g., 768-D)                 | Logits (4-D for 4 classes)    |
| Metric           | Cosine similarity                    | Softmax + argmax              |
| Training         | Unsupervised                         | Supervised (fine-tuned)       |
| Interpretability | General similarity                   | Categorical probability       |
| Thresholding     | Manual (e.g., 0.8)                   | Confidence-based (prob > 0.5) |
| Use cases        | Semantic search, clustering          | Explicit classification       |
| Speed            | Needs multiple reference comparisons | One forward pass              |


**Intuitive Analogy**
| Analogy        | Embedding Model                                        | Classifier Model                                    |
| -------------- | ------------------------------------------------------ | --------------------------------------------------- |
| How it behaves | Measures “how close” two meanings are in general space | Decides “which bucket this text belongs to”         |
| Example        | “Are these two texts semantically alike?”              | “Is this text a problem, solution, topic, or year?” |
| Mental model   | Semantic **map** of the world                          | Decision **boundary** separating categories         |


**Practical Impact in our Case**

| Aspect       | Old (Cosine)                   | New (Classifier)            |
| ------------ | ------------------------------ | --------------------------- |
| Endpoint     | `multilingual_e5_base_service` | `finetuned_model_inference` |
| Output shape | 768-dim embeddings             | 4-class logits              |
| Evaluation   | Similarity threshold           | Softmax probability         |
| Code         | `cosine_similarity()`          | `softmax → argmax()`        |
| Use case     | Clustering, retrieval          | Direct labeling in pipeline |


Notebook : https://github.com/aswinaus/ML/blob/main/ADLS_Databricks_ApacheSpark.ipynb
<img width="831" height="417" alt="image" src="https://github.com/user-attachments/assets/f3fa2972-b16e-45f7-990a-0b858a9bbda7" />

The Classification Model Training explicitly uses distributed XGBoost , leveraging multiple nodes in the cluster for scalable training.
Distributed XGBoost training in Databricks can performed using PySpark with parameters like num_workers to specify parallelism.
This enables efficient handling of large sharepoint data and faster model training times.

Notebook : ADLS_AzureSynapse_ApacheSpark.ipynb

<img width="929" height="704" alt="image" src="https://github.com/user-attachments/assets/b357d7e6-25df-45bd-a438-621f1be6ccf2" />

- Azure Blob Storage is the underlying object storage service.
- ADLS Gen2 extends Blob Storage with hierarchical namespace, fine-grained security and optimizing big data analytics.
- Azure Synapse Analytics provides a unified analytics platform combining big data (Spark Pools) and data warehousing (SQL Pools).
- Apache Spark running inside Synapse or Databricks uses Hadoop Azure filesystem connectors to read and write data from ADLS/Blob storage.
- Hadoop components (like YARN as resource manager in HDInsight) enable cluster resource management for Spark jobs.

**Key Challenges at Scale (Millions of Lines)**

| Challenge                      | What It Means                                           | How to Solve It                                             |
| ------------------------------ | ------------------------------------------------------- | ----------------------------------------------------------- |
| **LLM API Rate Limits**        | We can't call the API millions of times per minute     | Use batching, backoff, and parallelization                  |
| **Token Limits per Prompt**    | Each LLM (e.g., GPT-4) has a max token limit (\~128k)   | Limit number/size of docs per batch                         |
| **Memory & Collection Limits** | `.collect()` pulls all data to driver (can crash Spark) | Avoid `.collect()`; use `mapPartitions`, `foreachPartition` |
| **Long Job Runtime**           | Serial execution would be slow for millions of rows     | Use Spark’s distributed processing with UDFs or partitions  |
| **Error Handling**             | LLM API calls can fail (timeouts, overuse)              | Add retries, logging, and failover logic                    |

---------------------------------------------------------------------------------------------------------------------------------------------------------------

**Why Use Spark DataFrames vs Just Python.**

**Scalability**
| Feature                          | Spark DataFrame               | Python Only                      |
| -------------------------------- | ----------------------------- | -------------------------------- |
| Multi-core/multi-node processing | ✅ Yes (distributed computing) | ❌ No (limited to single machine) |
| Handles 10M+ documents?          | ✅ Easily                      | ⚠️ Risk of OOM / slowness        |
| Retry/fault tolerance            | ✅ Built-in                    | ❌ Must handle manually           |

**Data Integration and Pipelines**
| Feature                          | Spark DataFrame               | Python Only                      |
| -------------------------------- | ----------------------------- | -------------------------------- |
| Multi-core/multi-node processing | ✅ Yes (distributed computing) | ❌ No (limited to single machine) |
| Handles 10M+ documents?          | ✅ Easily                      | ⚠️ Risk of OOM / slowness        |
| Retry/fault tolerance            | ✅ Built-in                    | ❌ Must handle manually           |


**Core Advantages of Apache Spark (Beyond Just Distribution)**
| Feature                                 | Why It Matters                                                                                               |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Distributed Computing**             | Yes, it's the biggest one. Enables processing of **gigabytes to petabytes** of data across a cluster.        |
| **Unified Data Processing Engine**   | Supports **batch**, **streaming**, **SQL**, **ML**, **graph**, and **structured data** — all in one engine.  |
| **In-memory Processing**             | Faster than MapReduce because it keeps intermediate data in memory (vs writing to disk).                     |
| **Optimized for Big Data Workflows** | Built-in fault tolerance, DAG optimization, task scheduling, and caching.                                    |
| **Rich SQL Support**                 | Spark SQL lets us run **SQL queries on big data**, with full ANSI compliance and integration with BI tools. |
| **Easy Integration**                 | Reads/writes from: Azure Data Lake Storage                                                                                           |


**With Spark:**
1) Load all file paths into a DataFrame
2) Distribute text extraction + cleaning
3) Run mapPartitions to batch + classify via LLM
4) Store structured output into Delta Lake or SQL
5) Entire pipeline is parallel, fault-tolerant, scalable

| Databricks                           |
| ------------------------------------ |
| ✅ Fully managed and autoscaling     |
| ✅ Integrated with ADLS, Key Vault    |
| ✅ Built-in job scheduler & alerts    |
| ✅ Built-in lineage, logs, dashboards |
| ✅ Collaborative notebooks + repos    |

**Strategy to handle documents with images**

Reads .docx, .pdf, .xlsx files from ADLS.
Extracts embedded images
Saves images to ADLS
Optionally preprocesses the images (resize / convert / compress)
Send each image to Azure AI Vision via REST
Store the JSON results into a Delta table.
or Azure SQL.

**Cost Optimization Tips**
Area	Suggestions
Azure Document Intelligence	Batch process large documents to reduce API calls; avoid unneeded models (tables, signatures, etc.)
Azure OpenAI summarization / classification	Limit token size by summarizing first; or use extractive techniques + LLM on demand
Embeddings	Use OpenAI for best quality, or fallback to Hugging Face multilingual embeddings (e.g., sentence-transformers/paraphrase-multilingual-mpnet-base-v2)
Vector storage	Use hybrid indexes (keyword + vector) in Azure Search to reduce recall latency
GPU cost	Run summarization/classification in Databricks jobs with low-cost instances, and cache intermediate outputs (e.g., summaries)

**Based on summarizing a 200,000-word document via chunking**

| Model             | Max Input Tokens | Cost per 1K tokens (input + output)\*    | Approx Chunks for 200K words | Summarization Cost (est.) | Notes                              |
| ----------------- | ---------------- | ---------------------------------------- | ---------------------------- | ------------------------- | ---------------------------------- |
| **GPT-4o**        | 128,000          | \$5.00 / 1M input + \$15.00 / 1M output  | \~12–15 chunks               | ✅ **\$3–\$6**             | Best quality + efficiency          |
| **GPT-4 Turbo**   | 128,000          | Same as GPT-4o                           | \~12–15 chunks               | ✅ \~\$3–\$6               | Similar to GPT-4o, slightly slower |
| **GPT-4**         | 32,768           | \$30.00 / 1M input + \$60.00 / 1M output | \~40–50 chunks               | ❌ **\$20–\$40**           | High-quality but expensive         |
| **GPT-3.5 Turbo** | 16,384           | \$0.50 / 1M input + \$1.50 / 1M output   | \~80–100 chunks              | ✅ **\$2–\$4**             | Fast + cheap, lower quality        |
| **Davinci-003**   | 4,096            | \$0.02 / 1K tokens                       | \~350–400 chunks             | ❌ **\$10–\$20+**          | Legacy model, not efficient        |

**Assumptions:**
Each chunk is ~2,000–4,000 words (≈3,000 tokens)
We summarize each chunk to ~300 tokens
Total output is ~10,000 tokens per document
Costs include both input + output tokens

| Metric                                  | Estimate                                   |
| --------------------------------------- | ------------------------------------------ |
| 1 word ≈ 1.3–1.5 tokens                 | ⚠️ Approximate                             |
| 20 million words ≈ 28–30 million tokens | Let's use **30M tokens** as input estimate |


Ingestion steps (01–05) dont depend on each other can be run as parallel in Databricks Jobs.

               ┌────────────────────┐
               │01_ingest_word_docs │
               └────────────────────┘
               ┌────────────────────┐
               │02_ingest_xlsx_docs │
               └────────────────────┘
               ┌────────────────────┐
               │03_ingest_pptx_docs │
               └────────────────────┘
               ┌────────────────────┐
               │04_ingest_pdf_docs  │
               └────────────────────┘
               ┌────────────────────┐
               │05_ingest_msg_csv   │
               └────────────────────┘
                          │
                          ▼
               ┌────────────────────┐
               │06_redact_pii       │
               └────────────────────┘
                          │
                          ▼
               ┌────────────────────┐
               │07_local_E5_model   |
               │    classificaiton  |
               └────────────────────┘
                          │
                          ▼
               ┌────────────────────┐
               │08_embeddings       │
               └────────────────────┘
                          │
                          ▼
               ┌────────────────────┐
               │09_push_to_ai_search│
               └────────────────────┘
                          │
                          ▼
               ┌────────────────────┐
               │10_agent_query      │
               └────────────────────┘

Through the Databricks UI, we can define:

01–05 = parallel tasks

06–10 = sequential tasks

That way heavy ingestion steps scale out concurrently across clusters and the later processing stays ordered.

**Spark Driver Node vs Worker Nodes During Ingestion**

Spark Driver Node — The Brain

The driver node is responsible for:

Running the notebook/job code

Creating the SparkSession

Breaking code into logical stages and tasks

Sending tasks to the workers (executors)

Tracking progress and collecting results

The driver as the “orchestrator”.


**In short The driver:**

Parses the command

Determines the input file locations

Splits the files into chunks (partitions)

Plans a DAG (Directed Acyclic Graph) of tasks

Sends those tasks to the workers


**Spark Worker Nodes (Executors) — The Muscle**

The worker nodes (also called executors) are responsible for:

Reading the actual document data from storage

Executing transformations and computations on each data partition

Caching or persisting data in memory/disk if needed

Writing results back to storage (e.g., Delta Lake or Parquet)

Think of workers as “distributed data processors”.

**Direct Acyclic Graph of Tasks**


| Stage     | What It Does                                                               | Runs As      | Type of Spark Job                     |
| --------- | -------------------------------------------------------------------------- | ------------ | ------------------------------------  |
| **01–05** | Parallel document ingestion by file type ( PDF, MSG)                       | Python Tasks | Heavy I/O Spark read/write jobs       |
|           |                                                                            |              | Distributed Spark job with API calls  |             
| **06**    | PII redaction using Presidio detects and redacts PII Spark Job             | Python Task  | Transformation Spark job              |
| **07**    | Local E5 Model Classification                                              | Python Task  | Distributed semantic classification   |
|           |                                                                            |              | of large volumes of multilingual      |
|           |                                                                            |              | documents through local language model|
|           |                                                                            |              | Databricks(Training/Serving/Inference)|
| **08**    | Convert Classified content to embeddings                                   | Python Task  | Spark job text-embedding-3-large      |
| **09**    | Pull - Text to Embeddings Azure AI Search vector DB                        | AI Search    | Automatic data pull scheduled 20 mins |
| **10**    | Agent code to query Azure AI Search (retrieval +  enrichment)              | Python Task  | Light Spark/Driver job                |
| **11**    | Expose the Azure AI KH Search Index through API                            | Azure Python | AI Search + GPT4.0 finetuned          | 
|           |                                                                            | Function     |                                       |


**1. Databricks Job (Orchestrator / Launcher)**
A Databricks Job is the top-level execution unit — it can run:
A notebook
A Python script
A JAR or wheel
Define the tasks, dependencies, clusters, and parameters here.
Each task in the job triggers a Spark job if Spark is used inside it.
Think of it as:
"Run this workflow using Spark or Python"

**2. Spark Driver (Job Coordinator)**
When the Databricks Job runs Spark code (like reading a DataFrame), the Spark engine spins up a driver node.
The driver is responsible for:
Parsing the code
Building the logical plan (DAG)
Scheduling tasks
Tracking task progress and collecting results
It runs inside the cluster on a designated node.
Think of it as:
"Control tower of the Spark job"

**3. Spark Workers (Executors) (Task Runners)**
The worker nodes (executors) are where the actual data processing happens:
Reading and writing data
Executing .map(), .filter(), .groupBy() etc.
Writing Parquet files
Multiple executors can run in parallel on different nodes in the Databricks cluster.
Think of them as:
"Data crunchers doing the real work in parallel"

[ Databricks Job ]
       |
       ▼
[ Spark Driver ]
       |
       ▼
[ Spark Executors (Workers) ]


How They Interact (Example)

Imagine to run a Databricks Job that ingests PDFs:

df = spark.read.text("/mnt/data/*.pdf")

df = df.repartition(10)

df.write.parquet("/mnt/clean/pdf/")


Here is what happens:

| Layer               | What It Does                                                   |
| ------------------- | -------------------------------------------------------------- |
| **Databricks Job**  | Launches the script or notebook on a cluster                  |
| **Spark Driver Node**    | Parses code, builds DAG, creates 10 read/write tasks           |
| **Spark Worker Nodes (Executors)** | 10 executors read files, process them, and write Parquet files |

----------------------------------------------------------------------------------------------------------------------------------------------------------

**HashingTF explained** : 

Explain the HashingTF step with an example to make it clearer.

Recall the line: **hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=1000)**

The purpose of HashingTF is to take a list of words (like the output from the Tokenizer) and convert it into a fixed-size numerical vector. It does this using a clever technique called the "hashing trick" to avoid having to build a huge dictionary of all unique words.

Here is how it works with an example:

Imagine we have a very small vocabulary and numFeatures is set to a small number, say 5, instead of 1000 for simplicity. This means our output vector will have 5 "bins" or dimensions.

Let's say we have a document with the following words after tokenization: ["the", "cat", "sat", "on", "the", "mat"].

The HashingTF transformer will:

**Apply a hash function to each word:** A hash function takes a piece of data (in this case, a word) and converts it into a numerical value (an integer). The key property of a good hash function is that the same input always produces the same output, and different inputs are likely to produce different outputs (though collisions can happen).

**Map the hash value to an index:** The hash value for each word is then mapped to an index within the range of 0 to numFeatures - 1 (which is 0 to 4 in our example). This is typically done using the modulo operator (%). For example, index = hash_value % numFeatures.

**Increment the count at that index:** For each word in the document, the count in the corresponding index of the output vector is incremented.
Let's illustrate with our example words and hypothetical hash values and indices (remembering numFeatures=5):

"the": hash -> 12, index -> 12 % 5 = 2. Vector: [0, 0, 1, 0, 0]

"cat": hash -> 7, index -> 7 % 5 = 2. Vector: [0, 0, 2, 0, 0] (Collision! "the" and "cat" hashed to the same index)

"sat": hash -> 9, index -> 9 % 5 = 4. Vector: [0, 0, 2, 0, 1]

"on": hash -> 3, index -> 3 % 5 = 3. Vector: [0, 0, 2, 1, 1]

"the": hash -> 12, index -> 12 % 5 = 2. Vector: [0, 0, 3, 1, 1]

"mat": hash -> 11, index -> 11 % 5 = 1. Vector: [0, 1, 3, 1, 1]

So, the output "rawFeatures" vector for this document would be [0, 1, 3, 1, 1]. This vector represents the frequency of words in the document grouped into 5 bins based on their hash values.

**Why use the Hashing Trick?**

**Scalability:** It avoids building a large vocabulary dictionary, which can be very memory-intensive when dealing with millions of documents and a vast number of unique words.

**Speed:** Hashing is generally a very fast operation.

**Fixed Size:** The output vector size is fixed (numFeatures), regardless of the number of unique words in the entire dataset.

The main drawback is the possibility of hash collisions, where different words map to the same index. This can slightly reduce the model's ability to distinguish between words, but with a sufficiently large numFeatures (like the 1000 we used), the impact is usually minimal for many tasks.

After HashingTF, the rawFeatures vector goes to the IDF step, which will re-weight these frequencies based on how common words are across all documents.

------------------------------------------------------------------------------------------------------------------------------------------------------------

**Coefficient**:
Logistic Regression model a coefficient is a numerical value that represents the weight or importance of a particular feature (input variable) in predicting the outcome.

Here's a simple way to think about it:

Imagine we are trying to predict if someone will like a certain fruit based on two features: its sweetness and its color. A simple model might look something like this:

Likelihood of liking the fruit = (Coefficient for Sweetness * Sweetness Score) + (Coefficient for Color * Color Score) + (Intercept)

Coefficients: The numbers associated with Sweetness and Color are the coefficients.
A large positive coefficient for Sweetness would mean that the sweeter the fruit, the more likely someone is to like it.
A large negative coefficient for Color might mean that a certain color makes people less likely to like the fruit.
A coefficient close to zero would mean that the feature (Sweetness or Color) has little impact on the outcome.

In this Logistic Regression model, after training, the model will have coefficients associated with each of the features generated by the TF-IDF process. These coefficients indicate how much each word or term contributes to the model's prediction of the document's class.

The model learns these coefficients during the training process by analyzing the relationship between the features (the TF-IDF vectors representing the text) and the known labels of the documents. The goal is to find the set of coefficients that best allows the model to predict the correct label for each document.

------------------------------------------------------------------------------------------------------------------------------------------------

The inference code in Dual Encoder Model training notebook sends a document (doc) and a label (label) as input to the dual encoder model. This is different from a Large Language Model (LLM) with prompts in several ways:


**Input structure:** In an LLM with prompts, the input is typically a single string that contains the prompt, which may include the document and the label. In contrast, the dual encoder model takes two separate inputs: doc and label.

**Model architecture:** LLMs are typically designed as a single, large transformer model that generates text based on the input prompt. The dual encoder model, on the other hand, consists of two separate encoders: one for the document and one for the label.

**Training objective:** LLMs are typically trained on a masked language modeling objective, where the goal is to predict the next word in a sequence. The dual encoder model is trained on a contrastive learning objective, where the goal is to learn a similarity metric between the document and label embeddings.

**Inference:** During inference an LLM with prompts generates text based on the input prompt, whereas the dual encoder model computes a similarity score between the document and label embeddings.

**Similarities with LLMs and Prompts**

Despite these differences, there are some similarities between the dual encoder model and LLMs with prompts:

**Text-based input:** Both models take text-based input, although the dual encoder model takes two separate inputs.
Semantic understanding: Both models aim to understand the semantic meaning of the input text, although the dual encoder model focuses on learning a similarity metric between the document and label embeddings.

**Flexibility:** Both models can be fine-tuned for specific tasks and domains, allowing for flexibility in their application.

**Advantages of Dual Encoder Model**

The dual encoder model has some advantages over LLMs with prompts:
**Efficient computation:** The dual encoder model can be more computationally efficient than LLMs, since it only requires computing the embeddings for the document and label, rather than generating text.

**Improved accuracy:** The dual encoder model can achieve higher accuracy than LLMs for certain tasks, such as text classification and information retrieval, since it is specifically designed for these tasks.

**Interpretability:** The dual encoder model provides more interpretable results than LLMs, since the similarity score between the document and label embeddings can be easily understood and analyzed.

Overall, the dual encoder model and LLMs with prompts are both powerful tools for natural language processing tasks, but they have different strengths and weaknesses, and are suited for different applications.

**Here is a code snippet that shows how we can use the dual encoder model for inference:**

<img width="423" height="600" alt="image" src="https://github.com/user-attachments/assets/9a9f1283-2931-48e6-843e-06cb50df1ea5" />




-------------------------------------------------------------------------------------------------------------------------------------------------

**Quantization methods:**

**Dynamic quantization** – Easy, fast, good for NLP (e.g., BERT)

**Static quantization** – More accurate, requires calibration data

**QAT (Quantization-Aware Training)** – Most accurate, needs training


 №######Yet to be proved№#######
 
 step-by-step guide to deploy a quantized ONNX Transformer model (like BERT) to an Azure Machine Learning (Azure ML) real-time inference endpoint.


A quantized ONNX model (e.g., bert-base-uncased)

An inference endpoint on Azure ML (CPU-based, cost-efficient)

A working API we can call

Step 1: Setup Environment
🔧 Prerequisites
Azure subscription

Python environment (conda or venv)

Azure ML SDK

 Install Dependencies
pip install azure-ai-ml onnxruntime optimum[onnxruntime] transformers
 Login to Azure
az login
az account set --subscription "YOUR_SUBSCRIPTION_ID"
Step 2: Prepare and Quantize the Transformer Model
Let’s use bert-base-uncased as an example.

from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.exporters.onnx import main_export
from onnxruntime.quantization import quantize_dynamic, QuantType

import os

# Step 1: Export to ONNX
model_id = "bert-base-uncased"
output_dir = "./onnx_model"

main_export(
    model_name_or_path=model_id,
    output=output_dir,
    task="sequence-classification"
)

# Step 2: Quantize
model_fp32_path = os.path.join(output_dir, "model.onnx")
model_int8_path = os.path.join(output_dir, "model_quant.onnx")

quantize_dynamic(
    model_fp32_path,
    model_int8_path,
    weight_type=QuantType.QInt8
)
Step 3: Set Up Azure ML Workspace
Create a config file: config.json

{
  "subscription_id": "YOUR_SUBSCRIPTION_ID",
  "resource_group": "YOUR_RESOURCE_GROUP",
  "workspace_name": "YOUR_WORKSPACE_NAME"
}
Then connect:

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(DefaultAzureCredential(), path="./config.json")
 Step 4: Create Environment for ONNX Runtime
from azure.ai.ml.entities import Environment

onnx_env = Environment(
    name="onnx-runtime-env",
    image="mcr.microsoft.com/azureml/onnxruntime:latest",
    conda_file=None,
    description="ONNX Runtime Inference"
)

ml_client.environments.create_or_update(onnx_env)
Step 5: Write the Inference Script
Save as score.py:

import json
import numpy as np
import onnxruntime
from transformers import AutoTokenizer

model_path = "model/model_quant.onnx"
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
session = onnxruntime.InferenceSession(model_path)

def init():
    global session
    pass  # Model is already loaded

def run(raw_data):
    try:
        inputs = json.loads(raw_data)
        text = inputs["text"]
        tokens = tokenizer(text, return_tensors="np", padding=True, truncation=True)
        ort_inputs = {k: v for k, v in tokens.items()}
        outputs = session.run(None, ort_inputs)
        return {"logits": outputs[0].tolist()}
    except Exception as e:
        return {"error": str(e)}
Step 6: Register Model + Deploy
Register the quantized model
from azure.ai.ml.entities import Model

model = Model(
    path="onnx_model/model_quant.onnx",
    name="bert-quant-onnx",
    type="custom_model",
)

ml_client.models.create_or_update(model)
Create deployment config
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment

endpoint = ManagedOnlineEndpoint(
    name="bert-onnx-endpoint",
    auth_mode="key"
)

deployment = ManagedOnlineDeployment(
    name="default",
    endpoint_name=endpoint.name,
    model=model,
    environment=onnx_env,
    code_path="./",  # contains score.py
    scoring_script="score.py",
    instance_type="Standard_DS2_v2",
    instance_count=1
)

ml_client.online_endpoints.begin_create_or_update(endpoint).result()
ml_client.online_deployments.begin_create_or_update(deployment).result()
ml_client.online_endpoints.begin_assign_deployment(endpoint.name, "default").result()
🔍 Step 7: Test the Endpoint
endpoint = ml_client.online_endpoints.get("bert-onnx-endpoint")
print("Endpoint URL:", endpoint.scoring_uri)

# Get auth key
key = ml_client.online_endpoints.get_keys("bert-onnx-endpoint").primary_key

import requests

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {key}"
}

data = {"text": "Azure ML is great for deploying ONNX models!"}
response = requests.post(endpoint.scoring_uri, headers=headers, json=data)
print(response.json())
Done! we now have:
A quantized BERT ONNX model

Running on Azure ML as a scalable REST API

On CPU (cost-efficient) or GPU (if needed)

-------------------------------------------------------------------------------------------
**Hyperparameter Tuning:**
The supervised finetunning specific in the classificaiton problem which we had before computes the F1-macro and picks the model checkpoint when the F1-macro score is at the highest and it is restored at the end of training with one fixed hperparameter setup as in the code below. All these parameters are manually provided hyperparameters and none of these are tuned.

<img width="1172" height="291" alt="image" src="https://github.com/user-attachments/assets/b9c003f1-bcac-468d-ab10-61b54a60182f" />

**Ray Tune–powered hyperparameter tuning:**
This is one of the common type of hyperparameter tuning. Here we will retain SemanticTrainer, custom dataset and multi-label classification setup as we had before and run multiple training trials each with different hyperparameters with the F1-macro score to tune through which we pick the best hyperparameter combination and then save the model for serving.

Ray uses a resource-based scheduler where each task declares how many GPUs it needs.

**Step 1** — Ray detects GPUs on the node
**Step 2** — Tasks or Actors request GPU resources
  @ray.remote(num_gpus=1)
  def train_batch(...):
      ...
  This asks Ray:
  “Give this task 1 GPU.
**Step 3** — Ray schedules tasks according to GPU requests
If each task requests 1 GPU, Ray schedules:
Task 1 → GPU0
Task 2 → GPU1
Task 3 → GPU2
Task 4 → GPU3
→ 4 tasks run in parallel, each on its own GPU.
If a task requests 2 GPUs, Ray schedules:
Task A → GPU0 + GPU1
Then only 2 GPUs remain, so fewer tasks run.
If a task requests 4 GPUs, Ray schedules:
Task A → GPU0 + GPU1 + GPU2 + GPU3
Only 1 task runs at a time, but it gets all GPUs.

**In nutshell this is what Ray Tune will perform:**
Run multiple training trials each with different hyperparameters such as Learning rate, Batch size and epochs.

1) Report validation metrics (f1_macro) to Tune.
2) Pick the best hyperparameter combination.
3) Optionally save the best model.

**Ray Tune Trial Status Table:**

| **Status**     | **Description**                                | **Can Resume?** | **Typical Reasons / Notes**                                                        |
| -------------- | ---------------------------------------------- | --------------- | ---------------------------------------------------------------------------------- |
| **PENDING**    | Trial is created but has not started yet.      | Yes          | Waiting for GPU/CPU resources, queued by scheduler, cluster busy.                  |
| **RUNNING**    | Trial is actively training.                    | N/A             | GPU forward/backward passes, optimizer steps, validation, metrics reporting.       |
| **PAUSED**     | Trial is temporarily stopped but checkpointed. | Yes          | PBT/ASHA paused it to allocate resources to stronger trials. Can be resumed later. |
| **TERMINATED** | Trial finished successfully.                   | No            | Reached stopping criteria (epochs/iterations) or scheduler ended it cleanly.       |
| **ERROR**      | Trial crashed due to failure.                  | No            | CUDA not found, GPU not assigned, OOM, Python exception, bad hyperparameters.      |

-------------------------------------------------------------------------------------------------------------------------------------------------------

**** Nuances of saving the checkpoint and Resuming the training from checkpoint and let HF restore optimizer/scheduler from the checkpoint folder: {checkpoint.path}") vs resume_from_checkpoint in Hugging Face's Trainer ****
**
resume_from_checkpoint in Hugging Face's Trainer it will restore the following:**

Model weights: The weights of the model will be restored from the checkpoint.

Optimizer state: The state of the optimizer, including the learning rate, momentum, and other hyperparameters, will be restored from the checkpoint. In this case, the optimizer is AdamW.

LR scheduler state: The state of the learning rate scheduler, including the current learning rate, warmup steps, and other hyperparameters, will be restored from the checkpoint.

Global step / epoch counters: The global step and epoch counters will be restored from the checkpoint, so the training will resume from the correct point.

RNG state: The random number generator state will be restored from the checkpoint, which ensures that the training will continue with the same random seed.

Training arguments snapshot: A snapshot of the training arguments, including the batch size, gradient accumulation steps, and other hyperparameters, will be restored from the checkpoint.


Python objects: Any Python objects that are not serialized in the checkpoint, such as custom datasets or data loaders, will not be restored.
External state: Any external state, such as the state of other models or external libraries, will not be restored.
When using PeftModel.from_pretrained, it only loads the model weights, but not the adapter state. The adapter state including the optimizer and scheduler, is stored in the trainer_state.json file in the checkpoint directory.

By using resume_from_checkpoint, we can ensure that the entire training state, including the adapter state, is properly loaded from the checkpoint.

**Files saved when a trained model is saved in local after a trained with optimal tuned hyperparameters**

config.json
model.safetensors
pytorch_model.bin
sentencepiece.bpe.model
special_tokens_map.json
tokenizer.json
tokenizer_config.json

The trainer_state.json file is not being saved because we are using PeftModel.from_pretrained to load the model, which does not include the trainer state.

When we use PeftModel.from_pretrained, it only loads the model weights and configuration, but not the trainer state which includes the optimizer and scheduler.

To save the trainer_state.json file we need to use the Trainer class from Hugging Face, which saves the trainer state along with the model weights.

In this case, we are using PeftModel to load the model and then we are creating a new Trainer instance with the loaded model. However, when we save the model
using model.save_pretrained, it only saves the model weights and configuration, but not the trainer state.

To fix this, we need to use the Trainer instance to save the model which will save the trainer state along with the model weights. We can do this by calling trainer.save_model instead of model.save_pretrained. And this needs to be done after the PBT iteration loop this will ensure that the final model state is saved after the PBT iteration loop completes.

**To illustrate this, consider the following example:**

**Training 1:** Model converges to a loss value of 0.6 after 100 epochs.
Save the training state (model weights, optimizer state, etc.).

**Training 2:** Resume the training from the saved state. The model starts with the same weights and optimizer state as where the previous training left off.
The loss value is recalculated from scratch, and it may not be exactly 0.6. However, the model is likely to start with a similar level of performance, and the loss value may be close to 0.6 (e.g., 0.59 or 0.61).

In summary saving the training state allows us to resume the training from where the previous training left off but the loss value itself is not directly saved or restored. The model will start with the same weights and optimizer state but the loss value will be recalculated from scratch.

-------------------------------------------------------------------------------------------------------------------------------------------------------

**Advanced Tuning**

**Population Based Training (PBT)**

For the tax-document multi-label classification problem the learning rate, batch size and regularization strength are highly sensitive and interact in non-obvious ways so we would target those hyperparameters first.

PBT is a hyperparameter optimization + model training method where:

**1) Many model “workers” (populations) train in parallel**
Each worker starts with different hyperparameters (Learning Rate LR, batch size, etc).

**Every few iterations and the population is evaluated**
Workers doing poorly:
1) STOP
2) COPY weights from the best-performing workers
3) MUTATE their hyperparameters (slightly change LR, batch size, dropout, etc.)
   
**Bad configurations evolve into good ones**
This mimics biological evolution:
1) Good solutions reproduce
2) Poor solutions die
3) Random mutations allow discovery of better hyperparameters over time

It is efficient than grid/random search because bad configs don’t waste compute for long.

**Population Based Training will help tax-document classifier in the following ways**

**The labels are multi-label and imbalanced**
PBT adapts LR and batch size to reduce overfitting.

**We will use a custom semantic regularization loss**
This increases sensitivity to hyperparameters for which PBT will help to stabilize it.

**For the relatively small dataset (100 samples)**
PBT helps avoid overfitting and finds gentler LRs → boosts F1.

**Ray Tune powered Hyperparameter Tuning with PBT with Ray Tune status:**
<img width="1237" height="495" alt="image" src="https://github.com/user-attachments/assets/e4b3c733-120a-4b7b-a83b-a0324e59db1d" />

------------------------------------------------------------------------------------------------------------------------------------------------

**Distributed training (DDP - Distributed Data Parallel/FSDP - Fully Sharded Data Parallel) inside each trial, Ray is still essential for everything else the workflow requires:**

1. Ray gives us CONCURRENT HYPERPARAMETER SEARCH (HPO) using ALL GPUs
Even without distributed training, Ray Tune lets us run:
•	4 trials in parallel
•	each on a dedicated GPU
•	each with different hyperparameters
This IS NOT POSSIBLE with HuggingFace Trainer alone.
Without Ray
HuggingFace can only train one configuration at a time, on a single GPU.
With Ray
The run 4 completely independent trainings, each exploring a different hyperparameter region.
That’s the whole purpose of:
•	PBT
•	Bayesian optimization
•	Hyperparam sampling
•	mutation + exploitation
We are using Ray not for data parallelism, but for search parallelism.
________________________________________
2. Ray Tune = Population-Based Training (PBT) — which HuggingFace Trainer CANNOT DO
The scheduler is:
pbt = PopulationBasedTraining(...)
This algorithm requires:
•	multiple parallel workers
•	random mutation
•	cloning best-performing checkpoints
•	replacing weaker trials mid-training
HF Trainer cannot do PBT by itself.
Ray is the only reason we can use PBT.
________________________________________
3. Ray manages GPU allocation better than Databricks
Databricks has no native GPU queue.
Ray gives us:
•	resources_per_trial={"gpu": 1}
•	guaranteed GPU isolation
•	trial scheduling
•	placement groups
•	resource-aware parallelism
If we try 4 parallel HF processes without Ray → they will fight for GPU 0, crashing instantly.
Ray eliminates that.
________________________________________
4. Ray Tune handles trial retries, logging, checkpointing
We get:
•	automatic retries
•	storage of all trial artifacts under /root/ray_results
•	TensorBoard dashboard
•	unified logs per trial
•	final best-trial selection
Using pure HF Trainer → we must manually script all this.
Ray gives it for free.
________________________________________
5. Ray workers are isolated MINIPROCESSES
This matters because:
•	each trial loads a fresh BGE-base model
•	each trial loads its own LoRA adapters
•	each trial reads its own dataset subset
•	each trial runs completely independently
This is a MASSIVE stability improvement over multi-threaded pure Python.
________________________________________
6. Ray makes autoscaling possible
Later, if we move to Ray on Kubernetes or Ray cluster:
•	We can run 50+ trials on 50 GPUs
•	perfect for model exploration
•	nothing changes in the script
Ray makes the workflow future-proof.
________________________________________
7. Ray Tune does not require distributed training
Ray is NOT a distributed training framework.
Ray gives us task-level parallelism, not gradient parallelism.
We can have:
✔ All trials run independently
✔ Each trial uses 1 GPU
✔ No distributed training
✔ Still full PBT correctness
That is exactly the setup we need.
________________________________________
Analogy
Think:
•	HuggingFace Trainer = the model trainer
•	Ray Tune = the scientist running many different experiments in parallel
Even if the model trains only on 1 GPU, the experiment scheduler (Ray) is still crucial.
________________________________________
So the answer: YES — we still need Ray.
Ray is not about distributed training —
Ray is about scaling the number of experiments efficiently.

**Early stopping alone is not enough**
Early stopping stops bad models but it does not improve hyperparameters.

**-------------------------------------Evaluation----------------------------------------------------**

AUC-ROC stands for Area Under the Receiver Operating Characteristic Curve. It's a metric used to evaluate the performance of a binary classification model.

**What does it measure?**

AUC-ROC measures the model's ability to distinguish between two classes (e.g., positive and negative, 0 and 1, etc.). It plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at different threshold settings.


**Interpretation:**

**AUC-ROC values range from 0 to 1, where:**

1: Perfect classification (all positive instances are correctly classified, and no negative instances are misclassified)

0.5: Random chance (the model is no better than a random guess)

0: Worst possible classification (all positive instances are misclassified, and all negative instances are correctly classified)

A higher AUC-ROC value indicates better model performance.

**How to interpret AUC-ROC values:**

0.9-1: Excellent classification performance

0.7-0.89: Good classification performance

0.5-0.69: Fair classification performance

0.4-0.49: Poor classification performance

0-0.39: Very poor classification performance


In the classification evaluation code, the AUC-ROC value is calculated using the roc_auc_score function from scikit-learn, which takes the true labels and predicted probabilities as input. The resulting AUC-ROC value provides an estimate of the model's ability to distinguish between the two classes.


# ROC - Receiver Operating Characteristic Curve

ROC curve is a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied. It is created by plotting the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings.

AUC-ROC curve is a graphical representation of the trade-off between the true positive rate (TPR) and the false positive rate (FPR) at different classification thresholds

<img width="794" height="550" alt="image" src="https://github.com/user-attachments/assets/276f5661-84d8-47a9-a511-f69494e8973b" />

<img width="708" height="551" alt="image" src="https://github.com/user-attachments/assets/9f4ff23a-69f6-4338-852f-6b9c9ad31e0c" />



-----------------------------------------Evaluation----------------------------------------------------



-----------------------Cost Savings using local LLM for Classificaiton Problem--------------------------

1) Training / fine-tuning cost of the Local model on Databricks
2) Inference serving cost on a Medium GPU VM in Databricks
3) GPT-4.1 cost comparison for the same workload
4) True total cost over 1,000 documents


(These are the public reference prices Microsoft provided in 2024–2025.)

| Model       | Input (per 1K tokens)                          | Output (per 1K tokens)                          |
| ----------- | ---------------------------------------------- | ----------------------------------------------- |
| **GPT-4.1** | **$5.00 per 1M tokens → $0.005 per 1K tokens** | **$15.00 per 1M tokens → $0.015 per 1K tokens** |


So GPT-4.1 cost structure = same as GPT-4o
GPT-4.1 ≈ GPT-4o in token pricing

Token Count for a Typical 20-Page PDF
~7,000 input tokens
~300 output tokens
Total ≈ 7,300 tokens/document

Cost Calculation (GPT-4.1)
Cost per document
Input: 7,000 × $0.005 / 1,000 = $0.035
Output: 300 × $0.015 / 1,000 = $0.0045
✔ Total per document = $0.0395

Cost for 1,000 Documents (GPT-4.1)
$0.0395 × 1,000 = $39.50



Typical Training Workload

Dataset: 80k train + 10k val (≈ 90k samples)
Model: local model proposed (1.85B parameters)
LoRA + PBT training
Runtime: ≈ 1 hour per full training run (the logs confirm 58–60 min)

GPU pricing (2025 Databricks on Azure)

VM Type	GPU	Hourly DBU	Cost/hr
Standard_NC4as_T4_v3	1 × T4	6.5 DBU	~$5.00/hr

With $5/hour for training cost and to rerun PBT multiple times for 

5 runs it is $25.
For 100 runs it is $500 one time cost.

Inference Serving cost for Local Model on Databricks

Databricks Model Serving Example Pricing (2025)

Medium 2×T4 ~$4/hr

Inference Speed (the benchmark)

Local Model forward pass ≈ 5–10 ms per document with 1,000 docs = 10 seconds (batching: <5 seconds)

So Cost to classify 1,000 docs = about $0.01 with VM uptime (because the endpoint must run at least 1 hour)

If we run the endpoint only during processing: Minimum endpoint cost: $4/hour
Effective per 1,000 docs: $0.01 compute + $3.99 idle overhead
But this is fixed cost, not per-doc.
If used continuously throughout the day, per-doc cost becomes almost zero.

Total Cost Summary in hours
Cost Component	Amount
Training (1 hr)	$5.00
Serving (Medium endpoint, 1 hr minimum)	$4.00
Inference compute for 1,000 docs	$0.01
Total BGE End-to-End Cost	≈ $9.01

**This includes everything — training + serving + inference.**

3. GPT-4.1 Cost for 1,000 Documents

As calculated earlier:

✔ Cost per document = $0.0395
✔ 1,000 documents = $39.50

Following is the cost per hours for 1000 documents

| Cost Type                    | Local BGE (Databricks) | GPT-4.1 API               |
| ---------------------------- | ---------------------- | ------------------------- |
| **Model Training Upfront cost**           | **$500.00**              | $0 (but no customization) |
| **Model Serving**            | **$4.00**              | $0                        |
| **Inference for 1,000 docs** | **$0.01**              | **$39.50**                |
| **TOTAL for 1,000 docs**     | **$9.01**              | **$39.50**                |


| Cost Type                            | Local BGE (Databricks, First Model) | GPT-4.1 API |
| ------------------------------------ | ----------------------------------- | ----------- |
| **Model Training (one-time)**        | **$500.00**                         | $0          |
| **24-hour GPU Serving**              | **$96.00**                          | $0          |
| **Inference for 1,000 docs**         | **$9.00**                           | **$39.50**  |
| **TOTAL for 1,000 docs (first run)** | **$605.00**                         | **$39.50**  |

BUT THIS IS NOT THE REAL SAVINGS (IMPORTANT) 
Training + Serving is a fixed cost — not per-document.
If we process:
10,000 docs
100,000 docs
1,000,000 docs

the Local Model Serving serving cost stays mostly the same, while GPT-4.1 cost grows linearly.



Example for 100,000 docs Model	Cost 
Local Model (train + serve = fixed)	--> $500(training 100 hours) + $96(24 hrs of Model serving) + Inference Cost approx $10 = $605 
GPT-4.1 Inference -->	$3,950

Here how the cost flips

| Documents per day | Local BGE Cost Per 1,000 (96/day + inference) |
| ----------------- | --------------------------------------------- |
| 1,000 docs/day    | $105.00                                       |
| 5,000 docs/day    | $26.00                                        |
| 10,000 docs/day   | $12.00                                        |
| 50,000 docs/day   | $2.40                                         |
| 100,000 docs/day  | $1.20                                         |

While the first 1,000 documents appear more expensive due to one-time training and GPU provisioning the cost amortizes rapidly.


At scale (10,000+ docs/day), local model reduces classification costs from ~$40 per 1,000 (GPT-4.1) down to ~$12 per 1,000 — a 70% cost reduction.

As volume grows cost approaches ~$1 per 1,000 documents, achieving 97–99% savings.

For Classifying 4 million documents over period of two months

| Cost Component                          | Amount      |
| --------------------------------------- | ----------- |
| Initial training                        | **$500**    |
| Monthly feedback fine-tuning (2 × $200) | **$400**    |
| Serving compute (2 × $2,880)            | **$5,760**  |
| Inference cost for 4M docs              | **$36,000** |
| **TOTAL (2 Months)**                    | **$42,660** |

| Cost Component                                      | Amount       |
| --------------------------------------------------- | ------------ |
| Training                                            | **$0**       |
| Serving                                             | **$0**       |
| Inference (4,000 batches × $39.50 per 1,000 docs)** | **$158,000** |
| **TOTAL (2 Months)**                                | **$158,000** |

GPT-4.1 cost: $158,000
Local Model cost: $42,660
Savings = $158,000 − $42,660 = $115,340 saved per 2 months

73% reduction in cost
Local inference is 3.7× cheaper than GPT-4.1 classification
No token cost, full security, reduced latency

**----------------**
Approox Cost saved for $1000 document -  $3345
**--------------**

“Our local fine-tuned model costs $605 end-to-end (Upfront training + serving + inference) compared to $39.50 using GPT-4.1 for classifying 1,000 documents.


At scale, Local Model provides over 400× cost reduction, especially when classifying tens of thousands of documents daily.”
-----------------------------------------Cost Savings using local LLM for Classificaiton Problem-------------------------------------------------------------


-------------------------------------------------------------------------------------------------------------------------------------------------------
Training dataset
{"doc_id": "c6cfce94-4c21-421a-aefe-841739b28bbb", "doc_text": "Cybersecurity practices: Multi-factor authentication is enforced for privileged and administrative accounts consistently. Least privilege access policies restrict lateral movement and reduce attack surface. Endpoint detection and response tools flag suspicious process spawn chains promptly. Patch cycles remediate high-severity vulnerabilities according to prioritization policies. Secrets management rotates keys, audits vault access events, and eliminates hardcoding. Network segmentation isolates critical systems from general traffic and external exposure. Phishing simulations train staff to recognize deceptive content and report incidents. Backup encryption protects restore points and ensures integrity during recovery. Threat modeling reviews attack surfaces before major releases and architectural changes. Incident playbooks define containment, eradication, and recovery sequences with clear roles. Our advice is based on current tax legislation and subject to change. See Art. 13 CITA and relevant Kluwer commentary. Board resolution dated 2022-03-11 is attached.", "label_name": "tax_problem", "label_text": "A document that discusses a tax issue, error, dispute, challenge, or risk requiring attention. Often includes facts, circumstances, concerns, or problems that need resolution.", "target": 0, "soft_score": 0.0396, "pii_flag": 0, "pii_score": 0.0, "pii_types": []}

-------------------------------------------------------------------------------------------------------------------------------------------------------

**Soft and target refer to the labels or annotations associated with the training data.**

**Sample training data:**
{"doc_id": "c6cfce94-4c21-421a-aefe-841739b28bbb", "doc_text": "Cybersecurity practices: Multi-factor authentication is enforced for privileged and administrative accounts consistently. Least privilege access policies restrict lateral movement and reduce attack surface. Endpoint detection and response tools flag suspicious process spawn chains promptly. Patch cycles remediate high-severity vulnerabilities according to prioritization policies. Secrets management rotates keys, audits vault access events, and eliminates hardcoding. Network segmentation isolates critical systems from general traffic and external exposure. Phishing simulations train staff to recognize deceptive content and report incidents. Backup encryption protects restore points and ensures integrity during recovery. Threat modeling reviews attack surfaces before major releases and architectural changes. Incident playbooks define containment, eradication, and recovery sequences with clear roles. Our advice is based on current tax legislation and subject to change. See Art. 13 CITA and relevant Kluwer commentary. Board resolution dated 2022-03-11 is attached.", "label_name": "tax_problem", "label_text": "A document that discusses a tax issue, error, dispute, challenge, or risk requiring attention. Often includes facts, circumstances, concerns, or problems that need resolution.", "target": 0, **"soft_score": 0.0396**, "pii_flag": 0, "pii_score": 0.0, "pii_types": []}

**target:** This typically refers to a hard label, which is a binary or categorical label that indicates the presence or absence of a specific class or category. In the context of the tax problem, the target might represent a binary label indicating whether a document is related to a tax problem (1) or not (0).


**soft:** This refers to a soft label, which is a probability distribution over all possible classes or categories. Soft labels can be used to represent uncertainty or ambiguity in the labeling process. In the context of the tax problem, the soft label might represent a probability score indicating the likelihood that a document is related to a tax problem.


**target:** This is a hard label, which is a binary or categorical label that indicates the presence or absence of a specific class or category. In this case, the target label is 0, which means that the document is not considered a tax problem.


**soft:** This is a soft label, which is a probability score that indicates the likelihood of a document being a tax problem. The soft_score is 0.0396, which means that the model has a 3.96% confidence that the document is a tax problem.
The use of both target and soft labels allows the model to learn from both the hard labels (which provide a clear indication of whether a document is a tax problem or not) and the soft labels (which provide a more nuanced indication of the likelihood of a document being a tax problem).

In this specific example, the target label is 0, indicating that the document is not a tax problem, but the soft_score is 0.0396, indicating that the model has some confidence (although low) that the document might be related to a tax problem.


**Here's a breakdown of the possible scenarios:**

target = 1 and soft_score = 1.0: The document is clearly a tax problem, and the model is highly confident.

target = 0 and soft_score = 0.0: The document is not a tax problem, and the model is highly confident.

target = 1 and soft_score = 0.5: The document is a tax problem, but the model is uncertain (50% confidence).

target = 0 and soft_score = 0.5: The document is not a tax problem, but the model is uncertain (50% confidence).

-------------------------------------------------------**Scaled cosine similarity**----------------------------------------------------------

In the training code, the cos variable represents the cosine similarity between the document and label embeddings. The cosine similarity is a measure of how similar two vectors are, and it's calculated as the dot product of the two vectors divided by the product of their magnitudes.

The cosine similarity is then scaled by a factor alpha, which is a hyperparameter that's being tuned. This scaling is done to widen the range of the cosine similarity values, making it easier for the model to learn and differentiate between positive and negative pairs.

The scaled cosine similarity is used as the input to the BCEWithLogitsLoss function, rather than the raw cosine similarity values. This is because the BCEWithLogitsLoss function expects logits as input, which are the raw, unnormalized scores output by a model.


**Question :**  If alpha is used to scale the cosine similarity between two vectors in this case the doc text and lable text then what is the purpose of calculating contrastive loss?

In this case, the alpha is used to scale the cosine similarity between the document and label embeddings, which is then used to compute the BCE loss.

The contrastive loss, on the other hand, is used to encourage the model to learn embeddings that are close together for positive pairs (i.e., document and label are similar) and far apart for negative pairs (i.e., document and label are dissimilar).

The purpose of calculating the contrastive loss is to provide an **additional training signal that helps the model** to learn more informative and discriminative embeddings. By minimizing the contrastive loss, the model is encouraged to:

**Pull together positive pairs:** The model is encouraged to learn embeddings that are close together for positive pairs, which helps to improve the similarity between the document and label embeddings.

**Push apart negative pairs:** The model is encouraged to learn embeddings that are far apart for negative pairs, which helps to improve the dissimilarity between the document and label embeddings.

The contrastive loss is calculated as:

**contrast = (targets * (1 - cos).clamp(min=0) + (1 - targets) * (cos - margin).clamp(min=0)).mean()**

where cos is the cosine similarity between the document and label embeddings, targets is the target label (1 for positive pairs and 0 for negative pairs), and margin is a hyperparameter that controls the minimum margin between positive and negative pairs.

By combining the BCE loss and the contrastive loss, the model is trained to learn embeddings that are both informative and discriminative, which can improve the overall performance of the model.

Here is a code snippet that shows the calculation of the contrastive loss:

%python

# 4) Contrastive loss on raw cosine (no sigmoid)
contrast = (targets * (1 - cos).clamp(min=0) + (1 - targets) * (cos - margin).clamp(min=0)).mean()

Note that the contrastive loss is only calculated for the positive and negative pairs, and not for the neutral pairs (i.e., pairs with a target label of 0.5).

--------------------------------------------------------**Scaled cosine similarity**---------------------------------------------------------

-----------------------------------------**Using BCEWithLogitsLoss for numerical stability and efficiency**---------------------

BCEWithLogitsLoss is a loss function that's commonly used for binary classification problems, where the goal is to predict a probability of belonging to one of two classes.

In this case, the BCEWithLogitsLoss function is being used with the scaled cosine similarity values, which are not necessarily logits. However, the BCEWithLogitsLoss function is still a good choice here because of its numerical stability and efficiency.

The reason for this is that BCEWithLogitsLoss is equivalent to the BCELoss function when the input is passed through a sigmoid function. In other words, BCEWithLogitsLoss(x) == BCELoss(torch.sigmoid(x)). 

By using BCEWithLogitsLoss with the scaled cosine similarity values, the code is effectively applying a sigmoid function to the input, which maps the cosine similarity values to a probability range between 0 and 1. This is useful because the cosine similarity values are not necessarily probabilities, but the sigmoid function helps to interpret them as such.


**Using BCEWithLogitsLoss in this way has several advantages:**


**Numerical stability:** BCEWithLogitsLoss is more numerically stable than BCELoss because it avoids the need to compute the sigmoid function explicitly. This can help to prevent overflow or underflow issues when working with large or small input values.

**Efficiency:** BCEWithLogitsLoss is often faster than BCELoss because it can take advantage of optimized implementations that are specifically designed for working with logits.

**Flexibility:** By using BCEWithLogitsLoss with the scaled cosine similarity values, the code can take advantage of the flexibility of this loss function to work with different types of inputs.

----------------------------------------**Using BCEWithLogitsLoss for numerical stability and efficiency**----------------------------

**Calculating the training time**

6 trials x 2 epochs x 18.75(300 samples in dataset with 16 dataset per batch) batches per epoch = 225 batches per trial
225 batches per trial x 4 PBT iterations per trial = 900 batches per trial
900 batches per trial x 6 trials = 5400 batches
Assuming 1-2 seconds per batch, the total training time would be around 1-2 hours

**How is Ray tune - num_samples(trials) indicated through the logs in Ray Tune while training ?**

In this the three trials were completed and early stopping was triggered as there was no improvement in loss. 

**Logs:**

<img width="1201" height="86" alt="image" src="https://github.com/user-attachments/assets/a9746fa2-7f03-4b49-af11-5e007dc521ff" />


**Trial status: 3 TERMINATED** <-- Terminated status means Training completed successfully.

Current time: 2025-12-23 14:53:18. Total running time: 7min 51s
Logical resource usage: 2.0/36 CPUs, 1.0/1 GPUs (0.0/1.0 accelerator_type:A10)

+-------------------------------------------------------------------------------+
| Trial name                   status         iter     total time (s)      loss |
+-------------------------------------------------------------------------------+
| train_with_pbt_054b0_00000   TERMINATED        4            159.224   1.20927 |
| train_with_pbt_054b0_00001   TERMINATED        4            152.212   2.10058 |
| train_with_pbt_054b0_00002   TERMINATED        4            150.893   1.4184  |
+-------------------------------------------------------------------------------+

**In Population Based Training in the context of explore/exploit how do we infer that the exploit has happened during training?**

After completing certain number of trials within training PBT periodically ranks trials and performs exploit/explore:

**Exploit:** Copy weights & hyperparameters from best to worst ie. replace the losing trial's weights with the winning trial's weights.

**Explore(Perturb):** Mutate hyperparameters slightly for diversity ie. randomly mutate the copied hyperparameters using hyperparam_mutations ranges. example : "warmup": tune.uniform(0.05,0.20)

If ranking is unstable (due to noisy metrics or dependency issues) PBT can fail to converge ie. Divergence after Convergence would never happen as a result.


**Logs**

[PopulationBasedTraining] [Exploit] Cloning trial 967a0_00001 (score = -0.506585) into trial 967a0_00000 (score = -0.531692)


2025-12-23 15:06:38,891	INFO pbt.py:917 -- 

In this case PBT has perturbed the hyperparameters from trial 967a0_00001 with the best loss score of -0.506585 to the trial 967a0_00000 with worst score -0.531692 meaning the hyperparameter(as shown below) are copied over from best to worst and the checkpoint is created and saved after which the training continues.

[PopulationBasedTraining] [Explore] **Perturbed** the hyperparameter config of trial967a0_00001:

<img width="1286" height="290" alt="image" src="https://github.com/user-attachments/assets/5444a083-a683-4d23-8df6-2aa55b039af1" />


(train_with_pbt pid=10552) Checkpoint successfully created at: Checkpoint(filesystem=local, path=/root/ray_results/dual_encoder_pbt/train_with_pbt_967a0_00001_1_2025-12-23_14-56-40/checkpoint_000007)

**Side Notes**

**Early stopping**
early_stop_patience=5
early_stop_min_delta=0.001

**PBT**
max_pbt_iters=2
perturbation_interval=1

**Ray Tune config**
num_samples=2

Above will cause mutation and perturbation as the number early_stop_patience was set to five meaning five continuous iterations hence early stopping would take long period of time to complete

If we want mutation by PBT then make sure early_stop_patience=3. 

Early stopping
early_stop_patience=3
early_stop_min_delta=0.001


PBT
max_pbt_iters=2
perturbation_interval=1

Ray Tune config
num_samples=2

since max_pbt_iters = 2, early stop won’t matter here because the trial ends after 2 iterations anyway.

**Hugging Face behavior:**

•	If max_steps > 0, num_train_epochs is ignored

•	Training stops strictly at max_steps

Meaning in out training case

Each trainer.train() call runs for 200 steps, which is:
200 / 125 ≈ 1.6 epochs

And when calling trainer.train() inside a PBT loop:

for it in range(config["max_pbt_iters"]):  # max_pbt_iters = 4
    trainer.train()

Total training per trial
1.6 epochs × 4 PBT iters ≈ 6.4 epochs

------------------------------------------------------------------------------------------------------------------------------

**Additional Notes after Evaluation**

evaluation dataset samples  100

Accuracy: 1.0000. The proportion of correctly classified examples.

F1 Score: 1.0000. The harmonic mean of precision and recall.

Precision: 1.0000. The proportion of true positives among all predicted positives.

Recall: 1.0000. The proportion of true positives among all actual positives.

AUC-ROC: 1.0. The area under the receiver operating characteristic curve, which measures the model's ability to distinguish between positive and negative classes.

Cosine Similarities between doc embedding and label embedding : [0.4929605722427368, 0.39989757537841797, 0.413132905960083, 0.4945484399795532, 0.39989757537841797, 0.39989757537841797, 0.39989757537841797, 0.39989757537841797, 0.39989757537841797, 0.413132905960083, 0.4929605722427368, 0.4813384711742401, 0.413132905960083, 0.4813384711742401, 0.39989757537841797, 0.39989757537841797, 0.5004759430885315, 0.413132905960083, 0.5004759430885315, 0.5004759430885315, 0.413132905960083, 0.4929605722427368, 0.4929605722427368, 0.5004759430885315, 0.39989757537841797, 0.413132905960083, 0.39989757537841797, 0.413132905960083, 0.4813384711742401, 0.4945484399795532, 0.39989757537841797, 0.4929605722427368, 0.5004759430885315, 0.39989757537841797, 0.4945484399795532, 0.5004759430885315, 0.413132905960083, 0.5004759430885315, 0.4929605722427368, 0.413132905960083, 0.5004759430885315, 0.4813384711742401, 0.39989757537841797, 0.39989757537841797, 0.4929605722427368, 0.39989757537841797, 0.5004759430885315, 0.39989757537841797, 0.413132905960083, 0.39989757537841797, 0.39989757537841797, 0.4813384711742401, 0.413132905960083, 0.4813384711742401, 0.4813384711742401, 0.39989757537841797, 0.4945484399795532, 0.39989757537841797, 0.413132905960083, 0.39989757537841797, 0.4945484399795532, 0.39989757537841797, 0.4929605722427368, 0.5004759430885315, 0.5004759430885315, 0.413132905960083, 0.4929605722427368, 0.413132905960083, 0.4929605722427368, 0.413132905960083, 0.39989757537841797, 0.413132905960083, 0.4929605722427368, 0.39989757537841797, 0.413132905960083, 0.413132905960083, 0.4813384711742401, 0.4813384711742401, 0.39989757537841797, 0.5004759430885315, 0.4813384711742401, 0.5004759430885315, 0.4945484399795532, 0.5004759430885315, 0.413132905960083, 0.413132905960083, 0.4929605722427368, 0.4929605722427368, 0.4945484399795532, 0.4945484399795532, 0.413132905960083, 0.4929605722427368, 0.4813384711742401, 0.413132905960083, 0.4813384711742401, 0.413132905960083, 0.5004759430885315, 0.413132905960083, 0.39989757537841797, 0.4929605722427368]

Predicted Labels: [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1]
True Labels: [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1]
AUC-ROC interpretation:

0.9-1.0 : Excellent classification performance
0.7-0.89: Good classification performance
0.5-0.69: Fair classification performance
0.4-0.49: Poor classification performance
0.0-0.39: Very poor classification performance

# The F1 score is the harmonic mean of precision and recall for a class. When fine tuning a model the training objective is cross-entropy loss specifically in this case where we have multiple independent labels like problem, solution, tax type, tax topic and tax year the correct one is Binary Cross-Entropy(BCE) also can be called as Sigmoid + BCE loss which is the standard for multi-lable classificaiton and this is from where the gradient is computed and F1_macro metric is computed after each epoch (or batch) as a validation metric not as a loss like in RL where a reward signal directly drives optimization (e.g. in RLHF or GRPO), F1-macro is only used for monitoring and model selection - it does not produce gradients. It tells if the model is improving across all classes fairly.


AUC-ROC (Area Under the Receiver Operating Characteristic Curve) is indeed an evaluation metric used to assess the performance of a binary classification model. It plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at different classification thresholds, and the area under this curve represents the model's ability to distinguish between the positive and negative classes.

In the context of the code, AUC-ROC is used to evaluate the performance of the model in classifying documents as either related to tax issues or not. The AUC-ROC score ranges from 0 to 1, where:

0.9-1.0: Excellent classification performance
0.7-0.89: Good classification performance
0.5-0.69: Fair classification performance
0.4-0.49: Poor classification performance
0.0-0.39: Very poor classification performance

The AUC-ROC score is calculated using the roc_auc_score function from the sklearn.metrics module, which takes the true labels and predicted scores as input
------------------------------------------------------

**Metrics:**

**Accuracy:** 0.65 (or 65%) - This means that the model correctly classified 65% of the samples in the evaluation dataset.
F1 Score: 0.4615 - This is the harmonic mean of precision and recall. A higher F1 score indicates better performance. In this case, the F1 score is relatively low, indicating that the model is not performing well in terms of both precision and recall.

**Precision:** 1.0 - This means that all the samples that the model predicted as positive (i.e., related to tax issues) were actually positive. However, this is a bit misleading, as we'll see in the confusion matrix.

**Recall:** 0.3 - This means that the model only correctly identified 30% of the actual positive samples (i.e., samples related to tax issues).

**AUC-ROC:** 1.0 - This indicates that the model is able to perfectly distinguish between positive and negative classes. However, this is likely due to the fact that the model is heavily biased towards predicting negative classes (as we'll see in the confusion matrix).

**Confusion Matrix:**
The confusion matrix shows the number of true positives, false positives, true negatives and false negatives.

True Negatives (TN): 50 - The model correctly predicted 50 samples as not related to tax issues.
False Negatives (FN): 35 - The model incorrectly predicted 35 samples as not related to tax issues, when they actually were.
True Positives (TP): 15 - The model correctly predicted 15 samples as related to tax issues.
False Positives (FP): 0 - The model did not incorrectly predict any samples as related to tax issues.

**Classification Report:**
The classification report provides more detailed information about the model's performance.

Class 0 (not related to tax issues):
Precision: 0.59 - The model correctly predicted 59% of the samples that were not related to tax issues.
Recall: 1.00 - The model correctly identified all the samples that were not related to tax issues.

F1-score: 0.74 - The harmonic mean of precision and recall for this class.

Class 1 (related to tax issues):
Precision: 1.00 - The model correctly predicted all the samples that were related to tax issues (but this is because there were no false positives).

Recall: 0.30 - The model only correctly identified 30% of the samples that were related to tax issues.

F1-score: 0.46 - The harmonic mean of precision and recall for this class.

Overall, the model is performing in terms of recall for the positive class (related to tax issues). This means that the model is missing many samples that are actually related to tax issues. The high precision for the positive class is misleading, as it's due to the fact that there are no false positives. The model is heavily biased towards predicting negative classes, which is why the AUC-ROC score is 1.0. To improve the model's performance, we may need to adjust the threshold or explore other techniques to reduce the bias towards negative classes.

**Here's a step-by-step breakdown:**

Load Evaluation Dataset: The code loads a JSON file containing the evaluation dataset.
Encode Label: It encodes a label text ("Discussion of a tax issue or tax solution.") using a tokenizer.
Loop Through Examples: The code loops through each example in the evaluation dataset.
Forward Pass: For each example, it performs a forward pass through the model, passing in the encoded document text and the encoded label text.
Compute Cosine Similarity: It computes the cosine similarity between the document embedding and the label embedding.
Predict Label: The predicted label is determined by comparing the cosine similarity to a threshold (0.42 in this case).
Store Results: The true label, predicted score, and predicted label are stored in separate lists.
Compute Metrics: After looping through all examples, the code computes various metrics, including:
Accuracy
F1 score
Precision
Recall
AUC-ROC (Area Under the Receiver Operating Characteristic Curve)
Print Results: The code prints the computed metrics, as well as the predicted scores, predicted labels, and true labels.
Purpose
The purpose of this code is to evaluate the performance of a binary classification model in distinguishing between documents that discuss tax issues or solutions and those that do not. The model's performance is measured using various metrics, including accuracy, F1 score, precision, recall, and AUC-ROC.

Model
The model used in this code is a dual encoder model, which is a type of neural network architecture that consists of two separate encoders: one for the document text and one for the label text. The model computes the cosine similarity between the document embedding and the label embedding to predict the label.

Threshold
The threshold value (0.42) is used to determine the predicted label. If the cosine similarity is greater than the threshold, the predicted label is 1 (indicating that the document discusses a tax issue or solution), otherwise it is 0.

Evaluation Metrics
The evaluation metrics used in this code are:

Accuracy: The proportion of correctly classified examples.
F1 score: The harmonic mean of precision and recall.
Precision: The proportion of true positives among all predicted positives.
Recall: The proportion of true positives among all actual positives.
AUC-ROC: The area under the receiver operating characteristic curve, which measures the model's ability to distinguish between positive and negative classes.
These metrics provide a comprehensive evaluation of the model's performance in binary classification tasks.

AUC-ROC Interpretation
The AUC-ROC score is interpreted as follows:

0.9-1.0: Excellent classification performance
0.7-0.89: Good classification performance
0.5-0.69: Fair classification performance
0.4-0.49: Poor classification performance
0.0-0.39: Very poor classification performance
This interpretation provides a general guideline for evaluating the model's performance based on the AUC-ROC score.

Model Reliability and Testing

How has the risk of generalization errors been assessed and addressed in the context of this AI model? (required)

To assess and address generalization risk we trained the model and validated using tax document samples along with some transfer pricing tax cases to ensure coverage. Reliability across varied and different inputs from different languages was evaluated using cross validation with a 20 percent dataset and out of distribution testing that included noisy documents with minor portion of it with tax and incomplete documents. AUC ROC was used as the primary metric to measure true generalization performance and the final threshold was derived directly from the ROC curve. Overfitting risks were monitored through gaps between training and validation performance stability across folds and a proposed tax professional review of low frequency cases and continuous drift detection via cosine similarity monitoring during production has been implemented. Issues identified like as class imbalance, noisy document weakness and missing reference outputs have been remediated while future backlog items include automated KPI dashboards, concept drift adaptation with improved coverage for extremely rare forms and dynamic thresholding are in the pipeline for future implementation.

What considerations have been made to ensure the model's scalability, and how are scalability-related risks mitigated? (required)
To ensure scalability, the model and pipeline are designed to handle increased usage and larger tax documents through cloud ready, horizontally scalable components, including Databricks clusters that autoscale based on workload. lightweight BGE M3 inference optimized for local or containerized deployment. Latency and throughput have been validated through load and stress testing using high volume batches of documents, and results showed stable inference times and no degradation in classification accuracy under peak loads. The system mitigates scaling risks through asynchronous job processing, retry mechanisms, lineage based logging, and separation of compute for different document types and segregating the images through separate compute. And in order to tackle the future load we have backlog items which include automated scaling dashboards(Dashboard), predictive auto scaling triggers, and extended testing across multi region or hybrid deployments.


How does the model perform on out-of-distribution data, and what measures are in place to manage associated risks?
The model s performance on out of distribution (OOD) data is evaluated using intentionally varied test samples post training the model which includes noisy scans, pages with less graphical data along with incomplete scanned pdf tax forms which were not present in the original training dataset. The tests confirmed that while the model maintains stable performance within expected tax document patterns the confidence naturally decreases for unfamiliar layouts or degraded input. To manage OOD risks, the system uses cosine similarity based drift detection, confidence score thresholds along with  human in the loop review and JSON schema validation to flag incomplete or unsupported outputs. Fallback mechanisms ensure uncertain classifications are routed for Tax professionals review with additional highligts for those assets rather than returned as final IPs. Continuous monitoring of drift, confidence trends and human feedback scores drives iterative data improvements along with any edge case forms are added to training on a rolling basis to strengthen robustness over time. 

Describe the protections in place to defend against adversarial attacks and manipulation. (GenAI Only)
The model is protected against adversarial attacks through multiple layers of input validation, content safety control and prompt hardening strategies. All incoming documents and prompts are filtered through the Content Safety REST API endpoint to detect toxic, malicious or manipulative patterns before they reach the model. 
The system also enforces a strict prompt template rule set that instructs the model to never follow instructions embedded inside documents and to automatically strip or ignore fields resembling prompt injection attempts, including phrases like ignore previous ,  system prompt, exfiltrate and suspicious URLs or command style text.
Additional defenses include schema validation, anomaly detection based on confidence scores and confidence score and human in the loop escalation for uncertain or abnormal cases. The model has been evaluated for adversarial robustness using malformed, misleading and instruction laden documents and safeguards were confirmed to prevent jailbreak style behavior. Threat model considerations included prompt injection, data poisoning, malformed PDF manipulation and unauthorized instruction execution with risk mitigations captured in audit logs and lineage tables. While no external red team audit has yet been completed, internal testing is ongoing and backlog items include formal adversarial training evaluations and third party security reviews to further strengthen model resilience.

**Explainability**

How are explanations of AI system outputs provided to users? 

The system is designed for content based tasks such as document classification extraction summarization and safety filtering and does not perform autonomous decision making about individuals. For this reason detailed model internal explanations such as chain of thought or internal reasoning traces are not exposed to users.
Transparency is provided through clear labeling of AI generated outputs user visible metadata such as classification type confidence indicators and source document references where applicable and human in the loop review and approval workflows for sensitive outputs.

Who needs explanations (e.g., consumers, stakeholders, regulators)? 
Explanations of AI system outputs are primarily intended for internal stakeholders and oversight functions rather than end consumers.

**Which models are prioritised for explainability? **

Explainability is not prioritised for specific models because the system does not rely on predictive or decision making models that generate outcomes affecting individuals.
The AI models used are applied to content based tasks such as document classification extraction summarization and safety filtering where outputs are intended to support internal workflows rather than drive autonomous decisions. As a result explainability is addressed at a system and process level rather than at an individual model level. This includes documentation of model purpose input and output boundaries confidence indicators where applicable and human in the loop review controls.

**What types of explanations are produced? **

The system produces process level and outcome level explanations rather than model internal or algorithmic explanations.
Explanations include clear labeling of AI generated outputs contextual information about the task performed such as classification extraction or summarization confidence indicators where applicable and traceability to source documents or input data.
For sensitive use cases explanations are supplemented by human in the loop review audit logs and documented business rules that govern how outputs are generated and used.
Model internal reasoning explanations such as chain of thought or algorithm specific feature attributions are not produced because the system does not perform autonomous decision making about individuals and such explanations are not required for the intended use.

Robustness

How does system ensure robustness against adversarial attacks, outlier data, and off-topic queries? 

Protection against adversarial attacks is achieved through content safety filtering prompt injection detection strict input handling and controlled prompt construction that treats untrusted content as data rather than instructions. System prompts and outputs are constrained by defined schemas and safety policies to prevent unintended behavior.

Outlier data and anomalous inputs are handled through input validation confidence indicators monitoring and human in the loop review for sensitive or unexpected outputs. The system is designed to surface low confidence or unusual results for review rather than act on them automatically.

What performance metrics are used to evaluate robustness?

Metrics and indicators used include content safety and prompt injection detection outcomes input validation and rejection rates monitoring of off topic or unsupported queries logging of anomalous or outlier inputs confidence indicators where applicable and frequency of human in the loop escalations.
System level monitoring also includes audit logs error rates policy enforcement outcomes and review of false positives or false negatives related to safety controls. These measures are used to assess whether the system behaves consistently within its defined scope and handles unexpected or adversarial inputs safely.

How is sensitivity analysis performed (e.g., noise, perturbations)?

The system does not rely on predictive or decision making models with continuous numerical outputs and does not generate outcomes that affect individuals. AI capabilities are limited to content based tasks such as document classification extraction summarization and safety filtering within defined workflows.

What regularisation techniques are applied?

Data augmentation is used by expanding the training dataset with additional synthetically generated examples. Synthetic data is created using controlled python scripts based on representative sample data provided by the business team. This approach increases data diversity and improves classification robustness while preserving domain relevance and semantic validity.
Early stopping is applied as an explicit regularisation mechanism. Training progress is monitored using validation loss and convergence is considered to have stalled when the loss does not improve by at least early_stop_min_delta for a defined number of consecutive evaluations early_stop_patience. In this configuration early stopping is triggered when the validation loss does not decrease by at least 0.001 for more than three consecutive iterations.

How are vulnerabilities identified? 
In operation vulnerabilities are identified through monitoring of content safety outcomes audit logs error rates policy enforcement results and review of anomalies or unexpected system behavior. Findings from internal reviews security checkpoints and governance assessments are tracked and remediated as part of established risk management processes.

What mitigation strategies are in place for vulnerabilities?

A Sensor to Detect capability is implemented using content safety controls to identify prompt injection and adversarial input patterns. Detected events are logged along with the attack type and metadata in a delta table to support monitoring analysis and auditability.
A Sensor to Respond capability is implemented to automatically mitigate risk when repeated injection attempts are detected. If more than two sections or hunks within a document are flagged for prompt injection the system automatically prevents the document from proceeding to downstream asset generation. This mitigation is applied regardless of document type including tax documents that may otherwise contain valid problem or solution content.

**Data Quality**

How do we manage data quality and governance?

Sample data is provided by the business team and synthetic data is generated using controlled python scripts to augment coverage while preserving domain validity.

Data handling follows established data protection and security guidelines including access controls such as managing secrets in Key Vault securing ADLS connections using system managed identity private link configurations for ADLS Azure Search and Azure OpenAI controlled storage retention policies and comprehensive logging of data usage.
A system of record is maintained along with defined data retention controls to ensure consistency traceability and compliance across the data lifecycle.
Data quality issues anomalies and policy violations are identified through monitoring validation checks(PIM for secured Cloud resource access) and human in the loop review where applicable. Findings are logged reviewed and addressed as part of ongoing data governance and risk management processes.

Is exploratory data analysis (EDA) performed? When? 

EDA is performed to synthetic data generation process using controlled python scripts to ensure that augmented data remains aligned with domain expectations and does not introduce unintended artifacts or inconsistencies.

How do we perform data profiling, check distributions, and correlations? 
We performed Distribution checks mainly to assess class balance value ranges for binary classification problem and to make sure there is consistency between sample data provided by the business team and synthetically generated data created using controlled python scripts.

How do we handle missing data, outliers, duplicates, and class imbalance?

Missing data outliers duplicates and class imbalance are managed primarily during data preparation and exploratory data analysis prior to training. The training dataset is generated internally using controlled python scripts and business provided sample content. No public datasets or dynamic open source data sources are used.

What data quality dimensions and measurements are used?

We manage data consistency by verifying alignment between document content labels and derived attributes such as language tags and PII indicators across training validation and test splits. Relevance and representativeness are assessed by reviewing class coverage and distribution across labels including rare cases to ensure alignment with intended business use.

How do we ensure labelling consistency? 
Labelling consistency is ensured through deterministic rule based label generation embedded in the data synthesis pipeline Labels are assigned from explicit version controlled rules and validation logic Structural and content indicators are only applied when the corresponding text patterns are present and are automatically reconciled using rule checks and regex based detection We run post generation validation to confirm label to text alignment and reject or regenerate records that violate constraints Since labels are programmatically generated and not manually annotated inter annotator variability is not applicable All outputs are sanitized to the approved character set a-z A-Z 0-9 space - . _ : ( ) ; % & @ ? = /
Label schema included in the pipeline
tax_problem
tax_solution
tax_type
tax_topic
year
client_addressed - addressed to client partner external
internal_email - internal email
final_document - final version signature
draft_document - draft
long_document - long document indicator
short_email - 1 sentence trivial email
has_disclaimer - advisory disclaimer present
has_advisory_structure - intro executive summary analysis conclusion
has_sow_reference - references engagement letter agreement
has_citations - references footnotes citations
has_appendices - appendices present

How are discovered data quality issues resolved?

When an issue is detected  we analyse and  then the affected record is flagged and either corrected using deterministic rules or rejected and regenerated Root cause analysis is performed by reviewing the rule or generation logic. This is done through closely working with the testing team and development team.

------------------------------------------------------------------------

A machine learning engineer is configuring a hyperparameter search using SparkML's 'CrossValidator'. The goal is to optimize the 'maxDepth' parameter (for depths 5, 10, and 15) of a pre-defined 'DecisionTreeClassifier' instance named 'dt'. Which Python code snippet correctly uses the 'ParamGridBuilder' to define the search space?

The correct way to define the hyperparameter search space for maxDepth using Spark ML’s ParamGridBuilder is:

from pyspark.ml.tuning import ParamGridBuilder

paramGrid = (ParamGridBuilder()
             .addGrid(dt.maxDepth, [5, 10, 15])
             .build())

dt.maxDepth references the parameter from the existing DecisionTreeClassifier instance
.addGrid() specifies the values to try → [5, 10, 15]
.build() finalizes the grid for use in CrossValidator


-------------------------------------------------------------------------

A data science team is constructing a SparkML Pipeline to prepare high-dimensional sparse data for a distributed clustering algorithm. The pipeline must convert a categorical string column ('customer_segment') to a numeric format and aggregate all relevant inputs into a single feature vector. Given the component instances 'indexer', 'encoder', and 'assembler', which order correctly specifies the required sequence of transformations in the pipeline stages?

[indexer, encoder, assembler]

To properly handle a categorical string column like customer_segment and prepare it for clustering:

1. indexer (StringIndexer)
Converts string categories → numeric indices
Example: "Retail" → 0, "Enterprise" → 1
2. encoder (OneHotEncoder)
Converts indexed values → sparse binary vectors
Prevents the model from assuming ordinal relationships
3. assembler (VectorAssembler)
Combines:
Encoded categorical features
Any numerical features
Outputs a single features vector required by Spark ML algorithms

-------------------------------------------------------------------------

An ML Engineer needs to perform distributed training for a large-scale XGBoost model using the 'xgboost.spark' module on Databricks. The cluster has 16 worker nodes, configured with 4 CPU cores each. To maximize the utilization of all available Spark task slots across the cluster for training the 'SparkXGBClassifier', how should the parameter defining parallelism typically be configured?

To maximize utilization of all available Spark task slots, the key parameter in xgboost.spark is:

num_workers

The cluster has:

16 worker nodes
4 CPU cores per node

👉 Total available task slots =
16 × 4 = 64

So the correct setting is:

xgb = SparkXGBClassifier(
    num_workers=64
)
Explanation
In xgboost.spark, num_workers = number of parallel training tasks
Each worker maps to one Spark task slot
To fully utilize the cluster:
Set num_workers = total available cores across workers

------------------------------------------------------------------------------------

A transactional fraud detection application requires predictions based on the absolute latest version of the production model. The trained model is registered in Unity Catalog as 'prod.risk.fraud_model'. The model deployment pipeline updates the 'Champion' alias after successful validation. Which MLflow URI should the real-time inference pipeline use to ensure automatic referencing of the newly promoted model version?

The correct MLflow URI is:

models:/prod.risk.fraud_model@Champion

models:/ → indicates a model from the MLflow Model Registry
prod.risk.fraud_model → fully qualified model name in Unity Catalog
@Champion → model alias, not a fixed version

👉 This ensures:

The inference pipeline always uses the latest promoted “Champion” model
No code changes are needed when a new version is deployed
Automatic switching happens when the alias is updated

------------------------------------------------------------------------------------------------

A data scientist has developed a supervised learning model using scikit-learn and wants to apply it at scale for batch scoring a large Spark DataFrame in a production pipeline. The trained model artifact is stored via MLflow. Which statement(s) accurately describe the recommended approach for integrating this model efficiently into the distributed inference pipeline? (Select TWO correct options.)

✔️ 1. Use a Pandas UDF with mlflow.pyfunc.load_model
Load the model once per executor
Apply it in parallel using vectorized Pandas UDFs
Efficient for large-scale distributed inference

✔️ Example pattern:

import mlflow.pyfunc
from pyspark.sql.functions import pandas_udf

model = mlflow.pyfunc.load_model("models:/prod.risk.fraud_model@Champion")

@pandas_udf("double")
def predict_udf(*cols):
    import pandas as pd
    df = pd.concat(cols, axis=1)
    return model.predict(df)

df.withColumn("prediction", predict_udf(*feature_cols))

2. Use mlflow.pyfunc.spark_udf to create a Spark-native UDF
Simplest and recommended MLflow-native approach
Automatically handles:
Model distribution
Serialization
Efficient execution

✔️ Example:

from pyspark.sql.functions import col
import mlflow.pyfunc

predict_udf = mlflow.pyfunc.spark_udf(
    spark,
    "models:/prod.risk.fraud_model@Champion"
)

df.withColumn("prediction", predict_udf(*feature_cols))

  ----------------------------------------------------------------------

In the context of designing an ML pipeline in SparkML, an Estimator is conceptually distinct from a Transformer. Which definition accurately captures the core function of an Estimator instance in the pipeline workflow?

An Estimator is an algorithm that learns from data by fitting on a dataset and produces a Transformer.

In a Spark ML pipeline:

🔹 Estimator
Has a .fit() method
Takes a DataFrame as input
Learns parameters from data (training step)
Outputs a Transformer

✔️ Examples:

DecisionTreeClassifier
KMeans
StringIndexer
🔹 Transformer (for contrast)
Has a .transform() method
Applies a fixed transformation
Does not learn anything new

✔️ Examples:

OneHotEncoderModel
VectorAssembler
🔁 Pipeline Flow
Estimator --(fit)--> Transformer --(transform)--> Transformed Data

-------------------------------------------------------------------------------

A data engineering team sets up an automated retraining workflow using a Databricks Workflow (Job). The workflow includes three sequential tasks: T1 (Data Prep), T2 (Model Training), and T3 (Model Validation). T2 logs the model artifact and registers it to Unity Catalog, yielding a unique model URI. How can the model URI generated by Task T2 be reliably passed as an input parameter to the subsequent Task T3 (Model Validation) within the same Databricks Workflow execution?

Use dbutils.jobs.taskValues to set the model URI in Task T2 and retrieve it in Task T3.

Databricks Workflows provide a built-in mechanism for passing values between tasks in the same job run.

In Task T2 (Model Training)

After logging and registering the model:

dbutils.jobs.taskValues.set(
    key="model_uri",
    value=model_uri
)
🔹 In Task T3 (Model Validation)

Retrieve the value:

model_uri = dbutils.jobs.taskValues.get(
    taskKey="T2",
    key="model_uri",
    debugValue=None
)

--------------------------------------------------------------------

A Data Scientist is implementing a high-throughput image classification pipeline on Databricks. The task involves performing featurization using a pre-trained Keras model (TensorFlow backend) over millions of images stored in a Delta table. To optimize performance and leverage GPU resources, which approach is the most appropriate way to apply this single-node deep learning model at scale in a Spark DataFrame environment?

The most appropriate approach is:

Use a vectorized Pandas UDF (iterator-style) that loads the Keras model once per executor and performs batch inference on partitions, leveraging GPU-enabled clusters.

✅ Why this is correct

This pattern is ideal because it:

Scales across Spark partitions → parallel processing of millions of images
Loads the model once per executor → avoids repeated initialization overhead
Processes data in batches (vectorized) → efficient for deep learning
Leverages GPUs on each worker node when available
🔹 Recommended Implementation Pattern
from pyspark.sql.functions import pandas_udf
import pandas as pd

@pandas_udf("array<float>")
def featurize(iterator):
    import tensorflow as tf
    
    # Load model once per executor
    model = tf.keras.models.load_model("/dbfs/path/to/model")
    
    for batch in iterator:
        # batch contains a Pandas Series of image data/paths
        images = preprocess(batch)  # the preprocessing logic
        features = model.predict(images)
        yield pd.Series(list(features))
✅ Key Design Points
🔸 Iterator-style Pandas UDF
Uses iterator → processes batches instead of row-by-row
Minimizes serialization overhead
Best practice for deep learning inference in Spark
🔸 GPU Utilization
Each executor can:
Access a GPU
Run TensorFlow inference efficiently
🔸 Batch Processing
Deep learning models perform best on batches, not single rows

-----------------------------------------------------------------------------------------

A Financial Institution is building a real-time fraud detection endpoint using Databricks Model Serving. The model requires a feature representing the Euclidean distance between the current transaction location (provided in the API request) and the user's historical median location (stored in an online feature table). The distance calculation logic must be executed at the time of inference. Which entity should the MLOps engineer define to handle this dynamic calculation and seamlessly integrate it into the model serving endpoint?


Define an on-demand feature in the Databricks Feature Store (Feature Engineering).

The key requirement is:

A feature (Euclidean distance)
Computed at inference time
Using:
Request-time data (current transaction location)
Stored feature (historical median location)
🔹 Why On-Demand Features are correct

On-demand features are specifically designed for:

Real-time computation during model serving
Combining:
Online feature store values
Incoming request data

They allow us to:

Define transformation logic once
Automatically apply it during inference
Keep training and serving logic consistent
🔹 How it works conceptually
Retrieve stored feature:
User’s historical median location (from online feature table)
Combine with request input:
Current transaction location
Compute:
Euclidean distance
Pass result into model → prediction

The correct answer for the scenario described is:

A registered Unity Catalog Python UDF, referenced in a 'FeatureFunction' within a 'FeatureSpec' logged with the model.

--------------------------------------------------------------------------------------------------------------------------

A telecommunications company has billions of customer call records (unstructured text) stored in a Delta table in Unity Catalog. They need to categorize these calls (e.g., 'Billing Issue', 'Network Outage', 'Technical Support') using a state-of-the-art Generative AI model hosted by Databricks for periodic batch analysis. Which command is the most appropriate and scalable method for executing this large-scale inference directly within the data pipeline?

The most appropriate and scalable method is to use map_batches with the Databricks Generative AI model inside a Delta Live Table or Spark pipeline for batch scoring.

Recommended approach:
from databricks import generative_ai as genai

# Load the Generative AI model once per executor
model = genai.load_model("your_generative_ai_model_name")

def categorize_calls(batch_df):
    # Example: apply model to the 'call_text' column
    batch_df['category'] = batch_df['call_text'].apply(lambda text: model.predict_category(text))
    return batch_df

# Apply the function at scale with map_batches (vectorized)
categorized_df = calls_df.map_batches(categorize_calls)
Why this is best:
map_batches processes data in batches → efficient distributed execution
Loads model once per executor → avoids repeated initialization overhead
Scales easily to billions of rows in Delta tables
Keeps inference within Spark pipeline for seamless integration
Uses Databricks Generative AI hosted model → state-of-the-art NLP

SELECT call_text, ai_query('databricks-llama-4-maverick', CONCAT('Classify the following text: ', call_text)) AS category FROM customer_reviews_table;

-------------------------------------------------------------------

In a continuous deployment architecture, a Model Deployment pipeline is tasked with evaluating the newly trained 'Challenger' model against the current 'Champion' model before deciding which version should serve production traffic. After performing a successful A/B test comparison over two weeks, the pipeline determines the Challenger performs statistically better. How should the Model Deployment pipeline reflect this decision and ensure downstream inference pipelines use the superior model without requiring code changes or downtime?

Use the Unity Catalog Model Registry API to update the 'Champion' alias to point to the Challenger model version.

Explanation:
The Unity Catalog Model Registry supports aliases (like "Champion") that point to specific model versions.
Updating the "Champion" alias to the new Challenger version switches production traffic seamlessly.
Downstream inference pipelines reference the model via the alias (e.g., models:/model_name@Champion), so no code changes or downtime are required.
This approach aligns perfectly with continuous deployment and MLOps best practices for smooth model promotion.

-------------------------------------------------------------------------------------------

A Data Engineering team is tasked with setting up a highly reliable and performant pipeline on Databricks to continuously ingest millions of IoT sensor records saved hourly as semi-structured JSON files in cloud storage. The pipeline must handle schema evolution automatically and stream the raw data into a Bronze Delta table. Which component combination is essential for optimizing this ingestion process?

The correct choice is:

Lakeflow Declarative Pipelines utilizing Auto Loader configured with cloudFiles.format = "json" and publishing to a STREAMING TABLE.

Why this is the right combination:

Auto Loader is the Databricks component designed for incremental, scalable file ingestion from cloud storage, especially for millions of incoming JSON files.
It supports schema inference and schema evolution, which is exactly what we need for semi-structured IoT JSON data.
Writing into a Bronze Delta table through a streaming table gives reliable, fault-tolerant ingestion with Delta Lake benefits.

-----------------------------------------------------------------------------------------------------------

A Data Scientist is preparing a dataset for fine-grained demand forecasting. The raw input includes a categorical column 'day_of_week' (e.g., 'Monday', 'Tuesday'). They intend to use this feature, along with numerical features, to train a Random Forest Regressor using MLlib and require a feature representation that avoids implying ordinal relationships between days. Which sequence of standard MLlib feature transformers should be applied immediately before combining all features into the final input vector?


StringIndexer followed by OneHotEncoder.

Why:

day_of_week is a categorical string column.
StringIndexer first converts values like "Monday", "Tuesday" into numeric category indices.
OneHotEncoder then converts those indices into a one-hot vector, which avoids creating a false ordinal meaning between days.
After that, we use VectorAssembler to combine this encoded vector with the numerical features.

------------------------------------------------------------


A Data Science team has successfully trained an MLflow model that includes a complex custom preprocessing step (defined in a class 'CustomPreprocessor') and a scikit-learn XGBoost classifier. They need to deploy this entire logic unit (preprocessing + model) for low-latency real-time inference via a Databricks Model Serving endpoint. Which MLflow flavor deployment method is necessary to ensure the custom Python preprocessing code is correctly packaged, callable at inference time, and decoupled from the XGBoost binary artifact?

Creating an implementation that inherits from mlflow.pyfunc.PythonModel and overriding the predict() method to include the CustomPreprocessor logic.

Why:
A custom pyfunc model is the MLflow mechanism intended for packaging arbitrary Python inference logic, including preprocessing, postprocessing, branching, and framework-specific model loading, in a form that Databricks Model Serving can deploy. Databricks explicitly recommends pyfunc when the model requires preprocessing before calling the underlying model.

This approach keeps the XGBoost artifact separate from the custom Python code while exposing a single deployable inference interface. In a PythonModel, we typically place one-time loading in load_context() and per-request execution in predict(), which is exactly the pattern needed for low-latency serving with custom preprocessing.

--------------------------------------------------------------------------------

A development team is deploying distributed XGBoost training on a Databricks GPU cluster to handle a large dataset with highly sparse features. They want to ensure the training process is maximized for speed and efficiency using the 'xgboost.spark' estimator. Which three configurations are essential technical measures to optimize resource utilization and computation for this specific scenario? (Choose three.)


Set the SparkXGBClassifier parameter use_gpu to True.
Set the SparkXGBClassifier parameter num_workers to sc.defaultParallelism.
Set the SparkXGBClassifier parameters enable_sparse_data_optim=True and missing=0.0.

Why these three:

Databricks recommends use_gpu=True to enable GPU training with xgboost.spark.
For distributed training, num_workers should match the number of concurrent Spark tasks, and Databricks specifically recommends sc.defaultParallelism to use all Spark task slots.
For highly sparse features, Databricks documents that we should enable sparse-data optimization and set missing=0.0 when the features column contains SparseVector values.

-------------------------------------------------------------------------

A data team wishes to automatically identify financial fraud patterns by converting an existing large-scale, rule-based detection system into a machine learning model pipeline on Databricks. They decide to use the pre-existing rule results as training labels to train a Decision Tree Classifier using Apache Spark MLlib. They also need to integrate parameter tuning (Cross-Validation) and the feature preparation steps ('StringIndexer' and 'VectorAssembler') to ensure reproducibility. Which Apache Spark MLlib component is the most efficient choice to bundle the sequencing of these steps and apply the resulting complex logic to the training dataset?

The Pipeline class, which chains the feature transformers and includes the CrossValidator wrapper.

Why:

We need to bundle multiple stages into one reproducible ML workflow:
StringIndexer
VectorAssembler
DecisionTreeClassifier
CrossValidator for tuning
In Spark MLlib, the Pipeline is the component designed to sequence transformers and estimators into a single workflow that can be fit and then applied consistently to data.

---------------------------------------------------------------------------------------------

A Data Scientist has defined a SparkML Pipeline consisting of preprocessing steps (StringIndexer, VectorAssembler) followed by a LogisticRegression estimator (<code_example>lr</code_example>). To perform 5-fold cross-validation and tune <code_example>lr</code_example>'s <code_example>regParam</code_example>, which code snippet correctly configures the <code_example>CrossValidator</code_example>?

cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

Why:

estimator should be the entire pipeline, not just lr, so cross-validation evaluates preprocessing + model together.
Spark uses estimatorParamMaps, not paramGrid or paramMap, in CrossValidator.
evaluator must be passed separately.
numFolds=5 sets 5-fold cross-validation.

So the right option is the one with:

CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

-----------------------------------------------------------------------------------------------

A SparkML Pipeline incorporates an <code_example>RFormula</code_example> (aliased <code_example>rForm</code_example>) for feature transformation and a <code_example>LogisticRegression</code_example> model (aliased <code_example>lr</code_example>). A data scientist wants to optimize both the feature definition (<code_example>rForm.formula</code_example>) and the regularization parameter (<code_example>lr.regParam</code_example>). Which syntax correctly defines the hyperparameter combinations using <code_example>ParamGridBuilder</code_example>?

The correct syntax is:

ParamGridBuilder().addGrid(rForm.formula, list_a).addGrid(lr.regParam, list_b).build()

Why:

addGrid() takes a Param object, not a string.
rForm.formula and lr.regParam are the correct parameter references.
.build() produces the parameter map combinations for cross-validation or train-validation split.

So the right option is:

ParamGridBuilder().addGrid(rForm.formula, list_a).addGrid(lr.regParam, list_b).build()

--------------------------------------------------------------------------------------------

A SparkML <code_example>CrossValidator</code_example> job involving a large dataset runs slowly, logging frequent 'Shuffle Write' and 'Shuffle Fetch' activity. The computational complexity seems acceptable, but the I/O overhead is massive due to data movement between worker nodes for each fold. Which configuration setting is primarily responsible for tuning data distribution granularity during wide transformations and should be optimized?


spark.sql.shuffle.partitions

Frequent Shuffle Write and Shuffle Fetch indicate heavy data exchange during wide transformations.
spark.sql.shuffle.partitions controls the number of partitions used for shuffle operations, which directly affects data distribution granularity and shuffle overhead.
Tuning it can reduce excessive small tasks or overly large partitions during cross-validation on large datasets.

------------------------------------------------------------------------------------------------

A Data Scientist successfully uses <code_example>TrainValidationSplit</code_example> (<code_example>tvs</code_example>) to find the optimal Logistic Regression model contained within a Pipeline. The scientist persists the fitted object using <code_example>tvsFitted.write.overwrite().save(path)</code_example>. Which statement accurately describes the object saved and the subsequent loading process?

The entire TrainValidationSplitModel, including the best fitted PipelineModel, is saved and must be loaded using TrainValidationSplitModel.load(path) for seamless prediction.

Why:

After tvs.fit(...), the result is a fitted TrainValidationSplitModel.
Saving tvsFitted persists the model-selection result, which includes the best model found during validation.
Since the best model is inside a Pipeline, the saved artifact includes that fitted pipeline model as well, so preprocessing and prediction can be reused directly after loading.

Typical loading pattern:

from pyspark.ml.tuning import TrainValidationSplitModel

loaded_model = TrainValidationSplitModel.load(path)

Then we can use the best fitted model for inference through the loaded object.

So the right answer is:

The entire TrainValidationSplitModel, including the best fitted PipelineModel, is saved and loaded via TrainValidationSplitModel.load(path).

-----------------------------------------------------------------------------------------------------------------------

A Data Scientist is tuning a SparkML model using <code_example>CrossValidator</code_example> to detect credit card fraud (binary classification). Given the high cost associated with False Negatives (missed fraud cases), which metric should the <code_example>BinaryClassificationEvaluator</code_example> be primarily configured to maximize in order to prioritize minimizing False Negatives?

The best metric to maximize is:

recall (or truePositiveRate)

Why:

False Negatives mean actual fraud cases predicted as non-fraud.
Recall measures how many actual positive cases are correctly identified:
<img width="254" height="72" alt="image" src="https://github.com/user-attachments/assets/93983e49-1715-4476-90d1-69a4576ae66b" />

Maximizing recall directly helps minimize False Negatives, which is critical in fraud detection.

--------------------------------------------------------------------------------------

A long-running SparkML tuning job preprocesses a massive <code_example>DataFrame</code_example> (<code_example>raw_df</code_example>) using heavy transformations, yielding <code_example>transformed_df</code_example>. The subsequent <code_example>CrossValidator.fit()</code_example> process must repeatedly access this data for numerous trials and folds. To significantly reduce redundant computation, what action should be taken before calling <code_example>cv.fit(transformed_df)</code_example>?

Persist the intermediate DataFrame using transformed_df.cache()

Why:

CrossValidator.fit() will read the same transformed_df repeatedly across multiple parameter combinations and folds.
If the heavy preprocessing result is not cached or persisted, Spark may recompute the entire lineage many times.
Caching the transformed dataset avoids repeated expensive transformations and can greatly speed up model tuning.

In practice, it is often even better to materialize the cache right after:

transformed_df.cache()
transformed_df.count()

The key answer, though, is:

transformed_df.cache()

-----------------------------------------------------------------------------------------------------------

In a sophisticated Spark ML development workflow on Databricks, the Data Science team leverages the <code_example>MLflowClient</code_example> to find the best performing <code_example>TrainValidationSplitModel</code_example> from a long list of runs. Once found, the team extracts the actual fitted model from the tuning output. Which Spark MLlib class or concept must be utilized to correctly load the final trained model artifact embedded within the tuning result?

The desired model is extracted from the fitted tuning output by accessing tvsFitted.bestModel.stages and casting the appropriate stage element to LogisticRegressionModel.

Why:

A fitted TrainValidationSplitModel exposes its winning model through bestModel.
When the tuned estimator is a Pipeline, that bestModel is typically a PipelineModel, not directly a LogisticRegressionModel.
So to get the actual trained classifier, we inspect the pipeline stages and extract the final fitted stage, which is the LogisticRegressionModel.

So the key Spark MLlib concept involved is:

PipelineModel / pipeline stages extraction via bestModel.stages

-------------------------------------------------------------------------------------------------

When constructing a reusable and deployable machine learning workflow using SparkML Pipelines and the high-level MLlib framework, which of the following component types are fundamental elements defined within this structure? (Select TWO)

The two fundamental Spark MLlib Pipeline component types are:

Transformers
Estimators

Why:

Transformers take a DataFrame and return a transformed DataFrame.
Estimators are algorithms that are fit on data and produce a Transformer (for example, a trained model).

So the correct two are:

Transformers and Estimators

--------------------------------------------------------------------------

A Data Scientist uses the SparkML <code_example>LogisticRegression</code_example> estimator, which has the hyperparameter <code_example>elasticNetParam</code_example> that influences the mixing parameter for L1 (Lasso) and L2 (Ridge) penalties. If the goal is to favor a model with high regularization, simpler structure, and feature selection (Lasso), what value should be chosen for <code_example>elasticNetParam</code_example> during tuning?


1.0 (Pure L1/Lasso penalty)

Why:

In SparkML LogisticRegression, elasticNetParam controls the mix:
0.0 = pure L2 regularization
1.0 = pure L1 regularization
0.5 = equal mix of L1 and L2
If we want feature selection and a simpler sparse model, we favor L1/Lasso.

So the right choice is:

elasticNetParam = 1.0

------------------------------------------------------------------------------------------------------------------

A new fraud classification model is trained in a production workflow on Databricks. The subsequent task in the Databricks Workflow is Model Validation. The model artifact successfully loads, passes checks on format and required metadata, but fails a defined performance threshold check against a mandatory high-risk data slice. What action does the Model Validation pipeline take immediately upon this critical failure, according to the standard Databricks MLOps reference architecture?

The pipeline execution exits immediately, and alerts are configured via Workflows to notify users about the task failure.

In the Databricks MLOps workflow, after training, the model validation task checks the registered model artifact. Databricks states that if the model successfully passes all validation checks, it can be assigned the “Challenger” alias. If the model does not pass all validation checks, the process exits and users can be automatically notified.

So for a critical performance-threshold failure on a mandatory high-risk slice, the standard reference behavior is to stop the pipeline at validation rather than deploy, auto-retrain, or register it as a challenger/staging candidate.

---------------------------------------

A Databricks Workflow task is running an offline evaluation comparing a newly validated Challenger model against the current Champion model using a held-out dataset. The resulting AUC metrics confirm the Challenger outperforms the Champion. Assuming both models are registered in Unity Catalog, how are the precise comparison results typically recorded for detailed analysis within the Databricks platform?

The typical place for those detailed offline comparison results is MLflow Tracking.

In the Databricks MLOps reference workflow, the offline Challenger-vs-Champion comparison is performed on a held-out dataset, and Databricks says the workflow tracks the comparison results using the MLflow Tracking server. More generally, Databricks recommends MLflow to record metrics, parameters, tags, models, and other metadata for runs.

So the correct choice is:

The comparison results, along with model parameters and other artifacts, are tracked to the respective MLflow Tracking server associated with the execution run.

----------------------------------------------------------------------------------

An ML Engineering team needs to implement an A/B test for a high-traffic, low-latency scoring model using Databricks Model Serving. The requirement is to precisely route 50% of incoming production traffic to the new Challenger model version and the remaining 50% to the stable Champion version. Which mechanism is primarily utilized by Databricks Model Serving to facilitate this traffic splitting for continuous online evaluation?

Utilizing Model Aliases (Champion/Challenger) assigned to specific model versions combined with Model Serving endpoint traffic splitting functionality.

Databricks Model Serving supports a single endpoint serving multiple models and lets us configure a traffic split between the served entities, such as 50/50 for online A/B testing.

In the Databricks MLOps reference flow, models are often managed with “Champion” and “Challenger” aliases in Unity Catalog, and Databricks notes that we can create one endpoint with multiple models and specify the endpoint traffic split for Champion-vs-Challenger comparisons.

So the correct answer is the option combining:
Champion/Challenger aliases + Model Serving traffic splitting.

-----------------------------------------------------------------------------------------

A quality assurance team requires continuous monitoring of a deployed classification model to ensure fair performance across specific demographic segments (subpopulations). These segments represent less than 1% of the total inference traffic. Which best practice allows Databricks Lakehouse Monitoring to specifically evaluate model quality metrics for these small, predefined data slices post-deployment?

Defining custom metrics that specify the criteria for filtering inference data down to the required slice for performance evaluation.

But there is an important nuance: in Databricks Lakehouse Monitoring, the native best-practice mechanism for predefined subpopulations is actually slicing expressions, not custom metrics by themselves. Databricks lets us add “metric slicing expressions” so the monitor computes metrics for those subsets in addition to the full table. For example, an expression can create slices for a predicate and its complement, or one slice per unique value of a column.

Databricks also documents that, for profile creation, us can add custom metrics and slicing expressions in advanced options, and that the profile metrics are computed for each slice.

So, strictly speaking:

Best practice in Databricks: use slicing expressions for those demographic segments.
From the listed options: the second option is the nearest match.

-------------------------------------------------------------------------------------------------------

A credit scoring model must be deployed via a real-time Model Serving endpoint with a high Service Level Agreement (SLA) requiring the 99th percentile inference latency to be under 50ms. Which crucial form of pre-deployment testing specifically addresses evaluating whether the deployed model's system performance and infrastructure can satisfy this latency requirement?

Load Testing, comprehensively assessing performance, stability, and responsiveness under varying degrees of demand.

Why:

A 99th percentile latency SLA is about how the serving system behaves under realistic and peak traffic.
Load testing is the pre-deployment test used to measure whether the endpoint, model, and infrastructure can keep inference latency under the required threshold.
It helps validate tail latency such as p99 < 50ms, not just average response time.

--------------------------------------------------------------------------------------------------------------------

A data scientist needs to incorporate a custom SparkML model artifact, logged using mlflow.spark.log_model, directly into an existing structured streaming pipeline reading from a Delta table named customer_events_stream. The goal is to apply the model continuously to incoming data for real-time risk evaluation before writing results to a downstream table. Assuming the model is correctly registered as models:/risk_classifier/Production, which PySpark code snippet demonstrates the correct way to load and apply this model within the streaming pipeline?


model = mlflow.spark.load_model(model_uri="models:/risk_classifier/Production")
events_stream = spark.readStream.format("delta").table("customer_events_stream")
predictions = model.transform(events_stream)
predictions.writeStream.format("delta").start()

Why this one:

A model logged with mlflow.spark.log_model should be loaded with mlflow.spark.load_model. MLflow supports the Spark flavor for loading native Spark ML models.
Spark ML models are applied to a DataFrame using transform(), since Spark ML pipelines/models follow the Transformer interface.
A Structured Streaming source created with spark.readStream...table(...) is still a Spark DataFrame interface for transformations before writeStream.

--------------------------------------------------------------------------------------

During the Model Validation phase of the automated production MLOps workflow on Databricks, which checks are essential components of determining if a newly trained model is suitable for deployment and hence should proceed to the next stage in the Workflow? (Select TWO correct answers.)

Confirming the model artifact's format (for example, presence of model signature) and verifying required metadata for downstream deployment and inference.
Asserting that the model's statistical performance meets or exceeds a predefined threshold, potentially on targeted data slices.

Databricks describes model validation as including basic format and metadata validations plus performance evaluations on selected data slices when needed. If those validation checks pass, the model can proceed; if they fail, the process exits.

format/metadata validation and performance-threshold validation on overall or sliced data.

----------------------------------------------------------------------------------------------------

A pharmaceutical company needs to perform batch inference on 100 million records daily using a custom PyFunc deep learning model registered in Unity Catalog. To ensure the process is scalable and highly performant on a distributed cluster, which standard MLflow method should be used to integrate the Python model logic into the Spark data pipeline?

The standard scalable method is:

Use mlflow.pyfunc.spark_udf and apply it to the Spark DataFrame.

So the correct option is the one equivalent to:

loaded_udf = mlflow.pyfunc.spark_udf(spark, model_uri)
df_predictions = df.withColumn("prediction", loaded_udf(struct(df.columns)))

Why:

mlflow.pyfunc.spark_udf is the MLflow mechanism for running a PyFunc model distributed inside a Spark pipeline, which is the right pattern for large-scale batch inference on a cluster. Databricks documents batch inference on Spark DataFrames using a registered model, and MLflow documents spark_udf for applying Python-function models in Spark.
Converting 100 million rows to pandas, iterating with RDD map(), or sending bulk requests to an online serving endpoint are not the standard high-performance distributed Spark approach for this scenario.

mlflow.pyfunc.spark_udf(...) applied with withColumn(...).

--------------------------------------------------------------------------------

A telecommunications company uses Apache Spark Structured Streaming to process high-volume network events in near real-time. They need to continuously score these events using an existing Spark MLlib PipelineModel, which was logged using MLflow's Spark flavor. Which core method ensures that the model is applied correctly and efficiently within the streaming data flow?

The SparkML PipelineModel, being a Transformer, directly implements the distributed .transform() method, which should be applied immediately to the streaming DataFrame before writing to the sink.

Why:

A fitted Spark MLlib PipelineModel is a Transformer.
In Structured Streaming, we apply Spark ML models to the streaming DataFrame using model.transform(df_stream).
This keeps scoring distributed and native to Spark, which is the efficient pattern for high-volume streaming inference.

Why the others are not right:

Calling a serving REST endpoint for every micro-batch is not the standard Spark-native approach.
A row-by-row Python UDF is slower and unnecessary for an existing Spark MLlib model.
Collecting to a single RDD breaks scalability.
foreachBatch can be used in some workflows, but for an existing Spark ML Transformer, the core and correct method is still direct transform() on the streaming DataFrame.

So the answer is:

Use model.transform(df_stream) directly in the streaming pipeline.

-----------------------------------------------------------------------------------------------

An investment firm runs a critical daily portfolio valuation job requiring batch predictions against a petabyte-scale dataset stored in a Delta table. Given the massive scale and the non-critical latency (SLA of 8 hours), what is the primary consideration that favors using a dedicated Spark Batch inference job over deploying the model via a Model Serving REST API endpoint for this task?

Spark batch inference is the most cost-effective and resource-efficient solution for handling massive data volumes (high throughput) when ultra-low, sub-second latency is not required.

For Databricks, batch inference is the recommended pattern for large-scale prediction workloads, while Model Serving is built around REST-based online access. Databricks also documents a 16 MB per-request payload limit for Model Serving, which reinforces why a petabyte-scale daily valuation job is a poor fit for a serving endpoint and a natural fit for distributed Spark batch processing instead.

So the best answer is the cost/resource-efficiency and throughput advantage of Spark batch inference for massive offline workloads, not low-latency serving.

-----------------------------------------------------------------------------------------------

A model trained using Databricks Feature Engineering in Unity Catalog automatically handles feature lookup during scoring. A batch inference pipeline uses the FeatureEngineeringClient.score_batch() method to apply this model to a DataFrame. If the prediction dataset contains columns whose names match the features required by the model (e.g., 'age', 'income'), how does the Feature Engineering Client handle the feature retrieval for these specific columns?


It prioritizes the existing feature values in the input DataFrame and skips retrieving those specific features from Feature Store.

Databricks documents that by default a model packaged with feature metadata looks up features at inference time, but if we include a feature column in the DataFrame passed to FeatureEngineeringClient.score_batch(), that provided value is used instead. In their example, when the batch DataFrame includes account_creation_date, the API looks up only num_lifetime_purchases from Feature Store and uses the provided account_creation_date values for scoring.

So for columns like age or income that are already present in the prediction DataFrame, the client uses those local values and does not fetch those same features again from the offline store.

-----------------------------------------------------------------------------

A fraud team maintains a complex ML pipeline where the final prediction step requires combining pre-computed features from an online store with an 'on-demand' feature that must be calculated using a Unity Catalog Python UDF (e.g., distance). To ensure distributed scoring calculates and combines these features correctly during batch inference, which entity must encapsulate the feature combination and logic definition?


The Feature Store Model Lookup metadata, which links the Unity Catalog Python UDF using a FeatureFunction.

Databricks’ on-demand feature workflow says the feature-combination logic must be defined in the model’s feature lookup metadata during training by passing a FeatureFunction plus any FeatureLookup objects into create_training_set(). That metadata is then preserved when we log the model with fe.log_model(...), which is what allows score_batch() to automatically compute the on-demand UDF feature and combine it with looked-up features during distributed batch inference.

So the entity that encapsulates the feature combination and logic definition is not the Workflow job or a manually coded PyFunc wrapper. It is the model’s Feature Store / Feature Engineering lookup metadata, specifically the FeatureFunction definition bound into the training set and logged with the model.

-------------------------------------------------------------------------------------------------------

An insurance company uses a single Structured Streaming pipeline to read policy changes from Kafka and score the events immediately using a model trained on a prior version of the pipeline. Which MLflow URI syntax is required within the inference code to automatically load the specific model version currently designated for batch/streaming production use in Unity Catalog?

models:/<catalog>.<schema>.policy_model@Champion

Unity Catalog models use the full three-level name, and Databricks recommends aliases for deployment status. Databricks shows the inference URI format as models:/prod.ml_team.iris_model@Champion, and notes that workloads automatically pick up the new version when the alias is reassigned. It also states that stages are not supported in Unity Catalog, so .../Production is not the right pattern there.

So from the options, the correct one is:

models:/<catalog>.<schema>.policy_model@Champion

-----------------------------------------------------------------------------------

Which of the following capabilities or characteristics are primary drivers for choosing Apache Spark Structured Streaming inference (as opposed to batch inference) for a production application on Databricks? (Select ALL correct answers.)


The requirement to integrate complex change data capture (CDC) logic, handling updates and deletes, into the scoring data flow efficiently.
The need to process input data continuously or incrementally as soon as it arrives, typically targeting minutes-level latency or faster.
The business need to perform distributed, stateful computations (like moving averages or windowed aggregations) on the data before applying the model prediction.

Why these are the main drivers:

Structured Streaming is chosen when inference must happen on continuously arriving data, not on periodic static batches.
It is also the right fit when the pipeline must handle stream semantics, including CDC-style updates/deletes.
Spark Structured Streaming is especially useful when inference depends on stateful distributed transformations before scoring.

---------------------------------------------------------------

A retail application requires scoring 100 transactions per second with a 50ms P99 latency SLA for fraud detection. The prediction logic must incorporate a real-time computation of an on-demand feature (distance to last transaction location). Which solution is required to meet the low-latency requirement while integrating dynamic feature retrieval?

Deploy the model artifact using Databricks Model Serving, ensuring the model definition (FeatureSpec) includes the necessary Unity Catalog Python UDF for on-demand feature computation.

For a requirement like 100 TPS with a 50 ms P99 latency SLA, Databricks’ real-time serving stack is the appropriate pattern because Model Serving is designed for low-latency, high-availability online inference. Databricks also supports on-demand feature computation in Unity Catalog through FeatureSpec and feature functions/UDF-based definitions, so the distance-to-last-location feature can be computed as part of serving-time feature retrieval.

Model Serving + FeatureSpec with Unity Catalog Python UDF option.

-------------------------------------------------------------------------------------------------

A data pipeline must score a large Delta table (10 TB) containing transactional data using a trained Python Scikit-learn model logged via MLflow. Since the volume exceeds single-machine capacity, the inference must be distributed. Which implementation strategy ensures the most efficient scaling and maximized throughput for this batch inference job?

Load the model as a distributed function using predict_udf = mlflow.pyfunc.spark_udf(spark, model_uri) and apply it via df.withColumn('prediction', predict_udf(struct(features))).

Why:

mlflow.pyfunc.spark_udf is the standard MLflow approach for distributed batch inference on Spark DataFrames, which is exactly what we want for a 10 TB Delta table. It lets Spark execute inference across the cluster instead of forcing scoring onto one machine.
This is far more scalable than loading the model on the driver, using a row-by-row plain Python UDF, or converting the data to local NumPy/pandas objects. Those approaches do not fit a dataset of this size and would severely limit throughput.

So the correct option is the one using:

predict_udf = mlflow.pyfunc.spark_udf(spark, model_uri)
df = df.withColumn("prediction", predict_udf(struct(...)))

That is the most efficient and scalable batch-inference pattern here.

-------------------------------------------------------------------------------------------------------------------------------

An IoT platform needs to continuously score incoming sensor readings using a validated SparkML PipelineModel registered in Unity Catalog. The objective is continuous, streaming inference with a micro-batch latency target of 5 minutes. Assuming <code_example>events_stream</code_example> is the Structured Streaming DataFrame source, which code snippet demonstrates the correct and robust methodology for applying the distributed model to the stream?

model = mlflow.spark.load_model(model_uri)
predictions = model.transform(events_stream)
predictions.writeStream.start()

Why this is correct:

A validated SparkML PipelineModel logged with MLflow Spark flavor should be loaded using mlflow.spark.load_model(...).
A SparkML PipelineModel is a Transformer, so the correct way to apply it to a streaming DataFrame is model.transform(events_stream).
This keeps inference distributed and native to Spark Structured Streaming, which is the robust pattern for micro-batch scoring.

-------------------------------------------------------------------------------------------------------------

A media firm needs to classify the sentiment of 5 billion user comments stored in a Delta table. The classification must use a Databricks-hosted large language model (LLM) and be scalable to achieve high throughput within a scheduled batch window. Which inference technology is recommended for integrating this generative AI task efficiently into the unified data pipeline?

Use the native SQL function ai_query directly within a Spark SQL or DataFrame context to invoke the Databricks-hosted LLM model.

Databricks documents AI Functions as the built-in way to apply AI to data stored on Databricks, and specifically says ai_query() can be used for both generative AI and batch inference workloads. It is presented as optimized for batch inference and production workflows, which fits a scheduled, high-throughput classification job over a Delta table.

This is the best fit here because the requirement is to classify 5 billion comments in a unified data pipeline using a Databricks-hosted LLM. AI Functions let us invoke those hosted models directly from SQL over the Delta data rather than building custom driver-side loops, manual sharding, or billions of REST calls. Databricks also notes that Model Serving and AI Functions are tightly integrated for batch inference scenarios.

ai_query within Spark SQL / DataFrame context.

-----------------------------------------------------------

An online recommendation engine relies on highly dynamic features (e.g., last 5 minutes of clicks, computed in the Feature Store). The model artifact was trained and logged using the Databricks Feature Engineering Client, embedding the feature lookup metadata. Why is Databricks Model Serving the recommended platform for deploying this model for real-time inference (sub-100ms latency)?

It automatically handles the real-time lookup and joining of online feature values, ensuring consistency between training and inference time features, critical for low latency requests.

Databricks Model Serving supports automatic feature lookup for models logged with FeatureEngineeringClient.log_model, pulling required features from an online feature store at inference time.

Databricks also states that for real-time use cases, the serving endpoint uses the request’s entity IDs to look up pre-computed features from the online store and uses Unity Catalog lineage to resolve which features were used to train the model, which is what preserves training/inference consistency.

This is why Model Serving is the recommended platform for a recommendation engine with dynamic features and a sub-100 ms latency target.

-------------------------------------------------------------

An ML Engineer is evaluating production inference requirements. Which scenarios mandate the use of Apache Spark's distributed processing capabilities (either native SparkML or distributed UDFs) over Databricks Model Serving? (Select ALL correct options.)

Scoring a 100 million row Delta table weekly, generating bulk results that are consumed by a downstream BI reporting application.
Continuously applying a complex fraud detection rule set and scoring the augmented data using a gradient boosting model on events ingested directly via a Kafka topic.
Executing a large-scale, distributed matrix factorization (ALS) model on a 50 TB user-item interaction matrix for periodic recommendation scoring.

Why these require Spark:

Databricks recommends batch inference with Spark for large offline workloads over tables, which fits the 100 million row weekly scoring case.
Structured Streaming is the right choice for continuous event processing from sources like Kafka, especially when we need distributed transformations/rules plus model scoring in the stream.
A 50 TB ALS recommendation scoring workload is inherently a large-scale distributed Spark ML job, not an online serving pattern. Databricks positions Spark/Databricks Runtime for large batch and streaming workloads, while Model Serving is aimed at online or endpoint-style inference.

Why the other two do not mandate Spark over Model Serving:

Sub-10 ms P99 latency at 5,000 RPS is a classic Model Serving use case, not a Spark batch/streaming one.
Instant prediction for a single customer record from an external web request is also an online inference scenario best aligned with Model Serving.

----------------------------------------------------------------------------------

A pharmaceutical company is building a toxicity prediction model based on molecular graphs (complex, sparse data structure) that requires specialized model preprocessing code written in Python. This model needs to be run once daily on a batch of 50 million newly processed compounds. Since the data preprocessing is computationally intensive and benefits from distributed cores, which approach effectively scales this complex single-node prediction logic across the Spark cluster for batch inference?

Define the prediction workflow using mlflow.pyfunc.spark_udf and ensure the Spark configuration spark.sql.execution.arrow.maxRecordsPerBatch is optimized for vectorized data processing.

Databricks recommends mlflow.pyfunc.spark_udf(spark, model_uri) for distributed batch inference on Spark DataFrames, including custom Python model logic. That is the standard way to scale a single-node Python prediction workflow across the cluster for large batch jobs.

Databricks also recommends tuning spark.sql.execution.arrow.maxRecordsPerBatch to increase throughput by reducing UDF call overhead, as long as batches fit in memory.

So the best answer is the option with:

mlflow.pyfunc.spark_udf(...)
Arrow batch-size tuning via spark.sql.execution.arrow.maxRecordsPerBatch.

------------------------------------------------------------------------------------------------------

A data engineering team is running a resource-intensive feature calculation pipeline for over 200,000 unique time series in a distributed fashion. They use a standard Python function wrapped in a Pandas UDF, applied using .groupBy().applyInPandas() to ensure each time series is processed independently. What critical configuration step must be taken, specifically to prevent data skew from hindering performance during this distributed application?

Manually repartition the DataFrame by the distinct time series key columns before applying the .groupBy().applyInPandas() function.

Why:

applyInPandas() processes each group independently, so how data is partitioned across executors matters a lot.
Repartitioning by the time series key helps colocate rows for the same series and distributes groups more evenly, which reduces data skew and improves parallelism.
The other options either do not address skew directly or are incorrect for this scenario.

So the key step is:

df = df.repartition("time_series_key")

before:

df.groupBy("time_series_key").applyInPandas(...)

------------------------------------------------------------------------------------------------------------------

A team is fine-tuning a massive Large Language Model (LLM) on a Databricks multi-GPU cluster using PyTorch. During long training runs, they frequently encounter 'NCCL failure: remote process exited or there was a network error' messages, indicating communication issues between GPUs. Which configuration should the ML engineer adjust at the Spark cluster level to mitigate this specific network communication failure in the distributed training pipeline?

spark.executorEnv.NCCL_SOCKET_IFNAME to eth or eth0

Databricks documents this exact NCCL failure pattern for distributed PyTorch training and recommends setting the primary network interface via NCCL_SOCKET_IFNAME when multi-node GPU communication fails. On Databricks, that corresponds to setting the Spark cluster environment variable spark.executorEnv.NCCL_SOCKET_IFNAME to the correct interface such as eth0 (or eth depending on the environment).

So the correct option is:

Set the cluster's spark.executorEnv.NCCL_SOCKET_IFNAME to eth or eth0 to explicitly direct NCCL communication.

-------------------------------------------------------------------------------------------------------------------

A financial data team is preparing to run a large batch inference job using a scikit-learn model wrapped in an Arrow-optimized Pandas UDF applied via df.mapInPandas(). The source data is partitioned into 1000 Spark partitions. What critical performance benefit is gained by utilizing the Arrow/Pandas UDF framework over a traditional row-at-a-time Python UDF (non-vectorized) for this high-throughput scenario?

Arrow minimizes data serialization cost between the Python worker and the JVM by using columnar data transfer, significantly improving data throughput.

Why this matters:

Traditional Python UDFs process data row by row, which creates heavy JVM ↔ Python serialization overhead.
Arrow-enabled Pandas UDFs work on vectorized batches of data.
That batch-oriented, columnar exchange greatly reduces overhead and boosts throughput for large-scale inference.

So the correct choice is:

Arrow minimizes serialization cost via columnar transfer, improving throughput.

-----------------------------------------------------------------------------------------------

When training an ML model using SparkML's distributed K-Means algorithm on a dataset with millions of dense feature vectors, the job runs quickly but often fails with 'Out of memory' errors during the shuffle stage of aggregation. Which tuning action is recommended to address the memory pressure caused by the wide transformation without dramatically increasing the total number of physical files written?

Increase the cluster memory allocation and tune spark.sql.shuffle.partitions to a value approximately equal to the total number of CPU cores in the cluster.

Why:

The failure is happening during a shuffle-heavy wide transformation, so the main issue is shuffle partition sizing and executor memory pressure.
Too few shuffle partitions can make each partition too large, which increases per-task memory use and leads to OOM during aggregation.
Setting spark.sql.shuffle.partitions closer to the cluster’s total parallelism usually balances memory pressure without creating an excessive number of output partitions/files.
Increasing cluster memory helps support the aggregation workload.

Why the others are weaker:

Setting it blindly to 500 may help sometimes, but it is not the general tuning principle.
Lowering workers and partitions makes memory pressure worse.
df.cache() helps for recomputation, not specifically shuffle OOM.
Kryo can help serialization, but disabling shuffle service is not the right fix here.

So the best answer is:

Increase memory and tune spark.sql.shuffle.partitions to roughly the total CPU cores in the cluster.

-----------------------------------------------------------------------------------------------------------------------------

A team must train a complex deep learning model on a large dataset where training computation is split across multiple GPUs using the TorchDistributor interface. They observe inconsistent training times and poor GPU utilization. To maximize resource utilization during this distributed PyTorch training, which parallel data loading practice is specifically recommended within the Python environment?

Configure PyTorch's DataLoader class using high values for batch_size and num_workers to enable parallel data loading and batching.

Why:

In distributed PyTorch training, low GPU utilization is often caused by the GPUs waiting for data.
DataLoader with higher num_workers enables parallel CPU-side data loading/preprocessing.
Proper batch_size helps keep GPUs fed efficiently and improves throughput.

So the correct choice is:

PyTorch DataLoader with tuned batch_size and num_workers.

-----------------------------------------------------------------------------------------------------------------------

A Data Science team is developing a PySpark pipeline for large-scale financial modeling using a custom Python implementation of a Monte Carlo simulation. The computational cost per row is high. The team decides to use Ray on Databricks to distribute the workload using ray.data.from_spark(df).map_batches(custom_func). Which two core scaling benefits does using this combined architecture (Spark for DataOps + Ray for Compute) provide? (Select TWO correct options)

The two correct benefits are:

Spark efficiently manages large-scale data processing (ETL, filtering, aggregation) up to the point of computation, where Ray then handles the high computational intensity tasks (task parallelism).
The Ray Data API provides seamless, in-memory data transfer from the Spark DataFrame structure to Ray's distributed Dataset structure.

Why these two:

This is the core Spark + Ray pattern on Databricks: use Spark for DataOps / distributed data preparation, then hand off to Ray for Python-heavy compute parallelism like Monte Carlo simulation.
ray.data.from_spark(df) is specifically meant to bridge Spark DataFrames into Ray’s distributed data abstraction efficiently.

----------------------------------------------------------------------------------------------------------------

  



