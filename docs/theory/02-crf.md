# Linear-chain conditional random fields for PII tagging

## From spans to tags

A detector must output character spans. A sequence model labels tokens instead, so spans
are encoded with **BIO** tags: the first token of an entity of type X gets `B-X`, the
following tokens of the same entity `I-X`, every other token `O`. With privgate's 12 types
there are K = 25 tags. Decoding turns each `B-X I-X ...` run back into one span. A tag
sequence such as `O I-PERSON` is invalid, and privgate's decoder repairs it by starting a
new span. Tokens are `\w+` runs and single punctuation marks, so an entity boundary
inside a token cannot be represented. This tokenisation ceiling is measured in
docs/results-detection.md.

## The model

Let x = (x_1, ..., x_T) be a document's tokens and y = (y_1, ..., y_T) a tag sequence.
A linear-chain CRF (Lafferty, McCallum & Pereira, 2001) defines

    p(y | x) = exp( sum_{t=1..T} w . f(y_{t-1}, y_t, x, t) ) / Z(x)

    Z(x) = sum over all K^T sequences y' of exp( sum_t w . f(y'_{t-1}, y'_t, x, t) )

with y_0 a fixed start state. The feature vector f has two kinds of components:

- **state features** pair an observation with the current tag, e.g.
  `[previous word = "dr" and y_t = B-PERSON]`;
- **transition features** pair two consecutive tags, e.g. `[y_{t-1} = O and y_t = I-PERSON]`.

The model is discriminative: it never models p(x), so arbitrary overlapping features of
the whole input (neighbouring words, shapes, a regex detector's output) can be used.

## Training

The weights minimise the regularised negative log-likelihood of the training documents D:

    L(w) = - sum_{(x,y) in D} log p(y | x)  +  c1 * ||w||_1  +  c2 * ||w||_2^2

L is convex in w. Its gradient is "observed feature counts minus expected feature counts
under the model", and the expectation needs the marginals p(y_{t-1}, y_t | x). CRFsuite
minimises L with L-BFGS, a quasi-Newton method using a few recent gradient differences.
The L1 term is handled with OWL-QN, Orthant-Wise L-BFGS, because |w| is not
differentiable at 0. L1 sets many weights exactly to zero (a sparse model); L2 shrinks all
weights (stability). `c1` and `c2` are selected on the development set only.

## Inference

**Forward algorithm (for Z and marginals).** Let psi_t(i, j) = exp(w . f(i, j, x, t)) and
alpha_t(j) the summed score of all prefixes ending in tag j at position t:

    alpha_1(j) = psi_1(start, j)
    alpha_t(j) = sum_i alpha_{t-1}(i) * psi_t(i, j)
    Z(x)       = sum_j alpha_T(j)

This takes O(T K^2) operations instead of the O(K^T) of the naive sum. A symmetric
backward pass gives the marginals. In practice the computation is done in log space or
with scaling, to avoid overflow.

**Viterbi decoding (for the best tag sequence).** The same recursion with max instead of
sum,

    delta_t(j) = max_i delta_{t-1}(i) * psi_t(i, j),

with back-pointers to the argmax, gives argmax_y p(y | x) exactly, also in O(T K^2) time
and O(T K) memory. With K = 25 that is 625 operations per token.

## Why transitions help

A per-token classifier scores each tag in isolation. Nothing stops it from predicting
`O I-PERSON O`, or switching type in the middle of an entity (`B-EMAIL I-PERSON`).
Transition weights let the CRF learn that `O -> I-X` almost never happens. They also let
it learn that `B-PERSON -> I-PERSON` is likely, and that certain entity types follow each
other. Because decoding maximises the probability of the whole sequence, one token's
strong evidence (a title such as "Dr.") can carry into the next token's tag. Long entities
(addresses, organisation names) benefit most.

## Difference with per-token logistic regression

Multinomial logistic regression on the same features models each token independently:

    p(y_t | x) = exp(w_{y_t} . g(x, t)) / sum_k exp(w_k . g(x, t))

It has the state features of a CRF but no transition features, and it normalises per
token instead of per sequence. It is a CRF with every transition weight fixed at zero and
decoded greedily. Comparing the two on identical features (Milestone 4, RQ6) isolates
what the transitions contribute. The per-token model's invalid sequences have to be
repaired after the fact.

## Reference

J. Lafferty, A. McCallum and F. Pereira. *Conditional Random Fields: Probabilistic Models
for Segmenting and Labeling Sequence Data.* Proceedings of the 18th International
Conference on Machine Learning (ICML), 2001, pp. 282-289.
