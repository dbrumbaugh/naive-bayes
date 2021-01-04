# Naive Bayes Classifier
A Bayes Classifier is a probabilistic classifier that is based upon Bayes's Theorem, which is given by:
$$
\text{P}(A | B) = \frac{\text{P}(A)\text{P}(B | A)}{\text{P}(B)}
$$
Incidentally, this is a rather important mathematical result. If you only commit one thing from this course to memory, let it be this one.

We can apply Bayes's Theorem to the problem of classification by first stating the problem in the appropriate terms. Consider an individual data point to be classified. This point's known attributes can be represented by an argument vector, $\vec{a} = (a_0, a_1, a_2, a_3, \cdots, a_n)​$ and can belong to one of $k​$ disjoint classes, $C_k​$. 

We can calculate the probability, for each $C_k$, that this point falls into a given class given the values of its attributes. This can be represented using the conditional probability,
$$
\text{P}(C_k | \vec{a}) = \text{P}(C_k | a_0 \cap a_1 \cap a_2 \cap \cdots \cap a_n)
$$
Then, it is a simple matter to classify the point by calculating these probabilities, and selecting the class label for which the conditional probability is the largest.

Unfortunately, this particular statement of the problem is not very easy to solve, because the number of attributes can be quite large, making this conditional probability difficult to determine in practical situations. Thus, we restate the problem using Bayes's Theorem, in the hope of arriving at a formulation that is easier to solve.
$$
\text{P}(C_k|\vec{a}) = \frac{\text{P}(C_k)\text{P}(\vec{a} |C_k)}{\text{P}(\vec{a})}
$$
In this formulation, we now have three things to determine. However, the hope is that the determination of these parameters is easier than it originally would have been. Two of the three immediately lend themselves to an easy determination: the prior probability, $\text{P}(C_k)$ and the evidence, $P(\vec{a})$. In fact, the evidence term will remain constant for each class label for a given point, and, as we are simply examining the ordinal relationships between the posterior probabilities, we can safely ignore this term in our calculations. 

This does, however, level us with the prior probability, $\text{P}(\vec{a}|C_k)$ to contend with. At the surface, this looks to be no easier to deal with than the posterior probability. However, if we apply some algebra, and one critical assumption (from which Naïve Bayes draws its name), we can get even this term down to something easy to work with.

Recall from earlier in the course that the conditional probability is given by,
$$
\text{P}(A|B) = \frac{\text{P}(A \cap B)}{\text{P}(B)}
$$
If we substitute this into the numerator of Bayes's Theorem , we find:
$$
\text{P}(C_k) \frac{P(C_k \cap \vec{a})}{\text{P}(C_k)} = \text{P}(C_k \cap \vec{a})
$$
and, thus, 
$$
\text{P}(C_k|\vec{a}) \propto \text{P}(C_k \cap \vec{a})
$$
where we have dropped the evidence from consideration, due to its constant nature, thus reducing our equality to a proportionality. 

So now we have to contend with,
$$
\text{P}(C_k \cap a_0 \cap a_1 \cap \cdots \cap a_n) = \text{P}(C_k \cap \bigcap_{k=0}^na_k)
$$
We can rewrite this expression using the Probability Chain Rule, which states that,
$$
\text{P}(b\cap a_0 \cap a_1 \cap \cdots \cap a_n) = \text{P}(b | a_0 \cap a_1 \cap \cdots \cap a_n) \text{P}( a_0 \cap a_1 \cap \cdots \cap a_n) \\
\text{P}(b\cap\bigcap_{k=0}^na_k) = \text{P}(b |\bigcap_{k=0}^na_k) \text{P}( \bigcap_{k=0}^na_k)
$$
Incidentally, can you see how this operation follows directly from the definition of conditional probability that we've been using?

Because the intersection operator commutes, we can rearrange our join probability to put $C_k$ at the end, after all of the attributes.
$$
\text{P}(\bigcap_{k=0}^na_k \cap C_k)
$$
Now, we can apply chain rule. Doing this once will yield,
$$
\text{P}(\bigcap_{j=0}^na_j \cap C_k) = \text{P}(a_0 | \bigcap_{j=1}^na_j \cap C_k) \text{P}(\bigcap_{j=1}^na_j \cap C_k)
$$
However, we won't stop here. We can repeatedly apply the chain rule to the last term, until we have a product of conditional probabilities, with the final term being simply $\text{P}(C_k)$. 

This will result in an expression along the lines of,
$$
\text{P}(\bigcap_{j=0}^na_j \cap C_k) = \text{P}(a_0 |  \bigcap_{j=1}^na_j \cap C_k) \text{P}(a_1 |  \bigcap_{j=2}^na_j \cap C_k)\cdots \text{P}(a_n | C_k) \text{P}(C_k)
$$
Now comes the magic. We will assume that each attribute is conditionally independent of all other attributes. This is the assumption that gives this classifier the "Naïve" name. Obviously, this assumption is rarely true. However, even when it doesn't hold, classifiers based on it still manage to work remarkably well. The upshot of this assumption is that,
$$
\text{P}(a_i | \bigcap_{j=i+1}^n a_j \cap C_k) = \text{P}(a_i | C_k)
$$
Applying this assumption to our expression above results in the very tidy formula,
$$
\text{P}(a_0 | C_k)\text{P}(a_1|C_k)\cdots\text{P}(a_n|C_k)\text{P}(C_k) = \text{P}(C_k)\prod_{i=0}^n\text{P}(a_i|C_k)
$$
And, thus,
$$
\text{P}(C_k|\vec{a}) \propto \text{P}(C_k)\prod_{i=0}^n\text{P}(a_i|C_k)
$$
or, if we include the evidence term,
$$
\text{P}(C_k|\vec{a}) = \frac{1}{\text{P}(\vec{a})}\text{P}(C_k)\prod_{i=0}^n\text{P}(a_i|C_k)
$$
where $\text{P}(\vec{a})$ is a constant.

In order to apply this model, we simply seek the class label that maximizes its value,
$$
\hat{f}(\vec{a}) = \text{argmax}_k\text{ P}(C_k)\prod_{i=0}^n\text{P}(a_i|C_k)
$$
This technique is known as the maximum a posteriori (MAP) decision rule.

Of course, we have still neglected to discuss how, exactly, we are to find the conditional probabilities
$$
\text{P}(a_i|C_k)
$$
We could, technically, use our Laplace probability model and do it that way--that's definitely how we've done it up to now. But if we're working with attributes that are continuous, this will very quickly run into problems. What if the exact value we want doesn't exist in the training set? What can we do then?

We can (and this is sometimes done) simply bin our continuous domain into a few discrete buckets, and treat this using Laplace probability. However, it might be better to "graduate" from our simple Laplace probability, and move into more sophisticated probability models. We can use any of the common ones to calculate this conditional probability. For example, if $a_i$ is continuous and normally distributed, then we can apply a Gaussian Distribution,
$$
\text{P}(a_i | C_k) = \frac{1}{\sqrt{2\pi\sigma^2_k}} \exp\left(-\frac{(a_i - \mu_k)^2}{2\sigma_k^2}\right)
$$
where $\mu_k$ and $\sigma_k^2$ represent the mean and variance of the attribute $a_i$ across all points classified as $C_k$ in the training data.
