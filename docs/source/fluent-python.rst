Fluent Python
##########################################################################

* Chapter 1: Python data model

``__special_methods__`` are cool. They are called by the Python framework. You're not supposed to call them directly (except for `__init__` to call superclass constructor).

List of Special Methods:

#. ``__getitem__``: Random-access operator.
#. ``__len__``: Can be used by ``len(object)``.

Together, they allow the framework to do the following:

  #. Access any element with index (``object[key]``).
  #. Access n-th element from last with negative indexing (``object[-index_from_lat]``).
  #. Obtain random element using ``random.choice``.
  #. Slicing (``object[key1:key2]``) (TODO read more about slicing)
  #. Make the object iterator (``for _ in object:``)
  #. Generate a reverse iterator (``reverse(object)``)
