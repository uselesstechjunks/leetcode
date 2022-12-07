Python
#####################################

How Python works
******************************
I am trying to understand how Python works internally. In these notes, I'll capture the insights I found while trying to follow along the talk `Demystifying Python’s Internals <https://www.youtube.com/watch?v=HYKGZunmF50>`_ by Sebastiaan Zeeff at PyCon US for a deep-dive by implementing a new operator in Python. My plan is to eventually implement a haskell like Monad operator ``>>=`` (see `this <http://learnyouahaskell.com/a-fistful-of-monads>`_).

Setting up the system:
============================
* used the base Ubuntu docker image in my Linux machine and installed gcc, python3, make, git and vim.
* cloned cpython `git repo <https://github.com/python/cpython.git>`_.
* switched to branch 3.10 for my experiments for reproducibility.
* compiled cpython source which creates a binary named ``python`` inside the build dir.

Adding support for a ``pipe`` operator for chaining function calls
====================================================================================
The goal here is to implement an operator ``|>`` (similar to Linux ``|``) so that we can chain function calls like the following:

.. code-block:: python

    def sq(x):
      return x**2

    def double(x):
      return x*2

    42 |> sq |> double |> sq |> sq # essentially the same as sq(sq(double(sq(42)))))

When Python reads a source code, it first builds an abstract syntax tree (AST). This consists of two parts:

#. from the source code characters, it creates a sequence of tokens using a ``tokenizer``.
#. from the tokenized output, it then creates AST using grammar rules that are defined in the parser.

Building AST
-------------------------
#. Adding tokenizer support: 

Added a token named ``VBARGREATER`` with the token code to this file ``/cpython/Grammar/Tokens`` and ran ``make regen-token`` to regenerate the tokenizer.

.. code-block:: bash

    ....
    54 ELLIPSIS                '...'
    55 COLONEQUAL              ':='
    56 VBARGREATER             '|>'
    57
    58 OP
    59 AWAIT
    ....

Now I could see a difference in terms of how a source code is tokenized. Created a test file with the same python code as above and ran: ``python -m tokenize test/test.py``. Earlier, ``|`` and ``>`` were identified as separate tokens. Now, each instance of ``|>`` is treated as single token.

.. code-block:: bash

    # part of tokenizer output
    ...
    7,0-7,0:            DEDENT         ''
    7,0-7,2:            NUMBER         '42'
    7,3-7,5:            OP             '|>'
    7,6-7,8:            NAME           'sq'
    7,9-7,11:           OP             '|>'
    7,12-7,18:          NAME           'double'
    7,19-7,21:          OP             '|>'
    7,22-7,24:          NAME           'sq'
    7,25-7,27:          OP             '|>'
    7,28-7,30:          NAME           'sq'
    7,30-7,31:          NEWLINE        '\n'
    8,0-8,0:            ENDMARKER      ''

I also see that a bunch of other files have also been changed by this automatically.

.. code-block:: bash

    modified:   Doc/library/token-list.inc
    modified:   Grammar/Tokens
    modified:   Include/token.h
    modified:   Lib/token.py
    modified:   Parser/token.c

Let's dig deep into see what changes were made in each of these files and what these files are for.
