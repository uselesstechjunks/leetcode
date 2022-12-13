Fluent Python
##########################################################################

Chapter 1: Python data model
**************************************************

``__special_methods__`` are cool. They are called by the Python framework. You're not supposed to call them directly (except for ``__init__`` to call superclass constructor).

Special Methods for Collection Objects
============================================

#. ``__getitem__``: Random-access operator.
#. ``__len__``: Provides a functionality to obtain length, with ``len(object)``.

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
            
            
Special Methods for Numeric Objects
============================================

  #. ``__add__(self, other)`` implements ``self + other``.
  #. ``__mul__(self, other)`` implements ``self * other``.
  #. ``__abs__(self)`` implements ``abs(self)``.
  #. ``__repr__(self)`` implements a printable representation (enables ``print(object)``).
  #. ``__str__(self)`` implements a string representation (enables ``str(object)``).
  
      .. note::
          ``__repr__`` usually encodes a hint about the class as-well (e.g. ``MyClass(item_a=x, item_b=y)``) whereas ``__str__`` may represent it as ``[x,y]``. 
