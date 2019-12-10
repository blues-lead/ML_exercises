a)
$$
l(w) = \sum_{n=0}^{N-1}ln(1+e^{-y_nw^Tx_n})
$$
Where $w \in R^p, y \in \{-1;1\}, x \in R^p$


Find gradient w.r.t. "w" : $\frac{dl(w)}{dw}$

$$
\frac{dl(w)}{dw_1} = \sum_{n=0}^{N-1}\frac{e^{-y_nw^Tx_n}(-y_nx_1)}{1+e^{-y_nw^Tx_n}}
$$
$$
\frac{dl(w)}{dw_2} = \sum_{n=0}^{N-1}\frac{e^{-y_nw^Tx_n}(-y_nx_2)}{1+e^{-y_nw^Tx_n}}
$$
$$
\frac{dl(w)}{dw_i} = \sum_{n=0}^{N-1}\frac{e^{-y_nw^Tx_n}(-y_nx_i)}{1+e^{-y_nw^Tx_n}}
$$
$$
\frac{dl(w)}{dw} = \sum_{n=0}^{N-1}\frac{e^{-y_nw^Tx_n}}{1+e^{-y_nw^Tx_n}}(-y_n)
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
$$
b)
$$
l(w) = \sum_{n=0}^{N-1}ln(1+e^{-y_nw^Tx_n}) + Cw^Tw
$$
where $C\ge0$, regularization constant
$$
\frac{dl(w)}{dw} = \sum_{n=0}^{N-1}\frac{e^{-y_nw^Tx_n}}{1+e^{-y_nw^Tx_n}}(-y_n)
\vec{x}
 + 2C \vec{w}
$$