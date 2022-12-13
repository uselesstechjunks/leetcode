Fluent Python
##########################################################################

* Chapter 1: Python data model
======================================================

``__special_methods__`` are cool. They are called by the Python framework. You're not supposed to call them directly (except for `__init__` to call superclass constructor).

List of Special Methods:

#. ``__getitem__``: Random-access operator.
#. ``__len__``: Can be used by ``len(object)``.

Together, they allow the framework to do the following:

  #. Access any element with index (``object[key]``).
  #. Access n-th element from last with negative indexing (``object[-index_from_last]``).
  #. Obtain random element using ``random.choice``.

      .. code-block: python

          from random import choice

          item = choice(object) # returns a random item from object

  #. Slicing (``object[key1:key2]``) (TODO read more about slicing).
  #. Make the object iterable.

      .. code-block:: python
      
          for item in object:
            do_stuff(item)

  #. Generate a reverse iterator.
  
      .. code-block:: python
      
          for item in reverse(object):
            do_stuff(item)

  #. Enable querying for existance of an item by performing sequential scanning.
  
      .. note::
          If we implement a custom __contains__ function, then ``in`` would use that one instead)

  #. If we provice a custom ``item_ranker`` function, then we can also sort the items in the object using ``sorted`` interface.
  
      .. code-block:: python
          
          def item_ranker(item):
            return rank(item)
          
          for item in sorted(object, item_ranker):
            do_stuff(item)
            
            
