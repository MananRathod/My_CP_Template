class bitset():
    """
    Compact representation of a set of numbers (nonnegative integers)

    Internally, this class uses Python's arbitrary-precision integers to
    represent sets, where the *n*th bit of an integer indicates whether *n* is a
    member of the set.

    This class is intended to behave like the builtin ``set`` class, with
    members restricted to the nonnegative integers.  The bitset representation
    allows for some special behavior, such as naturally sorted iteration order,
    and the ability to represent the (infinite) complements of finite sets.

    This class is most efficient for relatively dense sets of small numbers.
    """

    def __init__(self, items=0, *, invert=False):
        """
        Initialize a new bitset.
        
        :param items: If a collection is provided, the bitset is initialized to
                      contain the elements of that collection.  If an int is
                      provided, the bitset's internal state is initialized to
                      that value.
        :param invert: If True, the bitset is inverted after the items parameter
                       has been processed.
        """
        if isinstance(items, int):
            self.bits = items
        else:
            self.bits = 0
            self.update(items)
        if invert:
            self.invert()

    def add(self, i):
        """ Add the given number to the bitset. """
        self.bits |= 1<<i

    def update(self, *iterables):
        """ Add the elements of the given iterables to the bitset. """
        for iterable in iterables:
            for item in iterable:
                self.add(item)

    def discard(self, i):
        """ Remove the given number from the bitset if it is present. """
        self.bits &= ~(1<<i)

    def remove(self, i):
        """
        Remove the given number from the set.

        :raises KeyError: If the element is not contained in the set.
        """
        e = 1<<i
        if not self.bits & e:
            raise KeyError(i)
        self.bits &= ~e

    def pop(self):
        """
        Remove and return the smallest number in the bitset.

        :raises KeyError: If the bitset is empty
        """ 
        if not bits:
            raise KeyError()
        obits = self.bits
        self.bits &= obits-1
        return (obits ^ bits).bit_length()-1

    def clear(self):
        """ Remove all numbers from the bitset """
        self.bits = 0

    def fill(self):
        """ Add all numbers to the bitset """
        self.bits = ~0

    def truncate(self, n):
        """ Remove all numbers greater or equal to n from the bitset """
        self.bits &= (1<<n)-1

    def copy(self):
        """ Return a new bitset containing the same numbers as the bitset """
        return bitset(self.bits)

    def invert(self):
        """ Replace the bitset with its complement """
        self.bits = ~self.bits

    def complement(self):
        """ Return a new bitset containing the complement of the bitset. """
        return bitset(~self.bits)

    def union(self, *others):
        """
        Return a new bitset containing the union of the bitset and all others.
        """
        bits = self.bits
        for other in others:
            bits |= other.bits
        return bitset(bits)

    def intersection(self, *others):
        """
        Return a new bitset containing the intersection of the bitset and all
        others.
        """
        bits = self.bits
        for other in others:
            bits &= other.bits
        return bitset(bits)

    def difference(self, *others):
        """
        Return a new bitset with numbers in the bitset that are not in the
        others.
        """
        bits = self.bits
        for other in others:
            bits &= ~other.bits
        return bitset(bits)

    def symmetric_difference(self, other):
        """
        Return a new bitset with elements in either the bitset or other but not
        both.
        """
        return bitset(self.bits ^ other.bits)

    def isdisjoint(self, other):
        """ Test whether the bitset has no elements in common with other. """
        return not (self.bits & other.bits)

    def issubset(self, other):
        """ Test whether every element in the bitset is in other. """
        return (self.bits & other.bits) == self.bits

    def issuperset(self, other):
        """ Test whether every element in other is in the bitset. """
        return (self.bits & other.bits) == other.bits

    def __bool__(self):
        return bool(self.bits)

    def __len__(self):
        if self.bits < 0:
            raise ValueError("Bitset is infinite")
        count = 0
        bits = self.bits
        while bits:
            bits &= bits-1
            count += 1
        return count

    def __iter__(self):
        """ Iterate over the elements of the set in sorted order """
        bits = self.bits
        while bits:
            bits_ = bits & bits-1
            yield (bits_ ^ bits).bit_length()-1
            bits = bits_

    def __contains__(self, i):
        return bool(self.bits & (1<<i))

    def __eq__(self, other):
        return not (self.bits ^ other.bits)

    def __le__(self, other):
        return (self.bits & other.bits) == self.bits

    def __lt__(self, other):
        return ((self.bits != other.bits)
                and (self.bits & other.bits) == self.bits)

    def __ge__(self, other):
        return (self.bits & other.bits) == other.bits

    def __gt__(self, other):
        return ((self.bits != other.bits)
                and (self.bits & other.bits) == other.bits)

    def __and__(self, other):
        return bitset(self.bits & other.bits)

    def __iand__(self, other):
        self.bits &= other.bits
        return self

    def __or__(self, other):
        return bitset(self.bits | other.bits)

    def __ior__(self, other):
        self.bits |= other.bits
        return self

    def __xor__(self, other):
        return bitset(self.bits ^ other.bits)

    def __ixor__(self, other):
        self.bits ^= other.bits
        return self

    def __sub__(self, other):
        return bitset(self.bits & ~other.bits)

    def __isub__(self, other):
        self.bits -= other.bits
        return self

    def __invert__(self):
        return bitset(~self.bits)

    def __str__(self):
        if self.bits > 0:
            return 'bitset({})'.format(set(self))
        elif self.bits < 0:
            if self.bits == -1:
                return '~bitset()'
            else:
                return '~bitset({})'.format(set(~self))
        else:
            return 'bitset()'

    def __repr__(self):
        if self.bits >= 0:
            return 'bitset({})'.format(bin(self.bits))
        else:
            return 'bitset(~{})'.format(bin(~self.bits))

# TODO: Type checks
# TODO: Support operator functions (union, intersection, etc.) on non-bitsets
# TODO: More efficient len()?
