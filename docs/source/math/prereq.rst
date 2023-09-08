#########################################
Pre-requisite
#########################################

Binomial and Multinomial Theorem
==============================================

(MISSISSIPPI problem): Number of ways to arrange the letters is :math:`\frac{11!}{1!4!4!2!}` (M=1,I=4,S=4,P=2).

.. note::
  * There are items of :math:`r` kinds, and within each kind they are indistinguishable. Say, we have :math:`k_i` items of :math:`x_i` kind, and so on. The number of ways of arranging a total of :math:`n` such items (i.e. :math:`sum_{i=1}^r k_i=n`)

    .. math::
      {n\choose k_1\cdots k_r}=\frac{n!}{k_1!\cdots k_r!}

Binomial theorem: 
-------------------------------
* For evaluating the expression :math:`(x+y)^n=(x+y)\cdots(x+y)` (:math:`n`-times), the process involves choosing either :math:`x` or :math:`y` from each of the :math:`n`-terms. 
* We can choose :math:`x` from 0 terms to :math:`n` terms. Every time we choose :math:`x` from :math:`k` such terms, we're left with no choice but to choose :math:`y` from  the remaining :math:`(n-k)` terms.
* The number of ways we can choose :math:`x` :math:`k`-times is :math:`{n\choose k}`. This is how many times we have :math:`x^k y^{n-k}`.
* Therefore, :math:`(x+y)^n=\sum_{k=0}^n {n\choose k} x^k y^{n-k}`.

.. tip::
  * :math:`\sum_{k=0}^n {n\choose k}=2^n` (setting :math:`x=1` and :math:`y=1`).
  * :math:`\sum_{k=0}^n (-1)^k {n\choose k}=0` (setting :math:`x=-1` and :math:`y=1`).
  * (Vandermonde Identity) :math:`{m+n\choose k}=\sum_{i=0}^k {m\choose i}{n\choose k-i}`.
  * :math:`{n\choose k}={n-1\choose k-1}+{n-1\choose k}`.

Multinomial theorem:
* For evaluating the expression :math:`(x_1+x_2+\cdots+x_r)^n=(x_1+x_2+\cdots+x_r)\cdots(x_1+x_2+\cdots+x_r)` (:math:`n`-times), the process involves choosing either :math:`x` or :math:`y` from each of the :math:`n`-terms. 
