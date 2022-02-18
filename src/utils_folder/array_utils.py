import numpy as np
from typing import List


def int_in_base_to_dec_int(base_array: np.ndarray, base:int) -> np.ndarray:
    """Takes an array of shape (y, x) or (x,), where the integers along dimension x
    represent entries of numbers to base base. If base=2, the sequence of integers
    gives a binary number. This method returns the corresponding integer in decimal.

    Args:
        base_array (np.ndarray): dtype=int, shape=(1, x) or (x,)
        base (int): base to convert from

    Returns:
        np.ndarray: shape(y, )
    """
    if len(base_array.shape) == 1:
        base_array = base_array[np.newaxis, :]
    assert np.max(base_array) < base, "The array is not in the given number system!"
    assert np.min(base_array) > -1, "The number array can only contain non-zero values!"
    base_multiples_array = calc_base_multiples_array(base_array.shape[0], base, base_array.shape[1])
    return np.sum(base_array * base_multiples_array, axis=1)


def calc_base_multiples_by_index(base: int, index_length: int) -> np.ndarray:
    """A number given in the specified base with the index length determines
    which multiples of the index is needed to return to decimal system.
    Example: base=3, index_length=4 returns array([[27, 9, 3, 1]])

    Returns:
        np.ndarray: shape = (1, index_length)
    """
    return np.array([base ** k for k in range(index_length-1, -1, -1)])[np.newaxis, :]


def calc_base_multiples_array(amount_of_num: int, base: int, index_length: int) -> np.ndarray:
    base_multiples = calc_base_multiples_by_index(base, index_length)
    return np.repeat(base_multiples, amount_of_num, axis=0)


def int_in_dec_to_base_in_array(dec_array: np.ndarray, base: int, repr_length:int) -> np.ndarray:
    if len(dec_array.shape) == 1:
        dec_array = dec_array[np.newaxis, :]
    base_repr_value = np.vectorize(np.base_repr)
    base_string_array = base_repr_value(dec_array, base=base)
    base_string_array = np.char.zfill(base_string_array, repr_length)
    return np.array([string_to_int_list(string[0]) for string in base_string_array.T])

def string_to_int_list(string_of_ints: str) -> List[int]:
    return [int(c) for c in string_of_ints]


def main():
    joint_action = np.array([[0, 0, 2, 1], [1, 0, 1, 2]])
    translated_array = int_in_base_to_dec_int(joint_action, base=3)
    print(translated_array)
    dec_int_array = int_in_dec_to_base_in_array(translated_array, base=3, repr_length=4)
    print(dec_int_array)


if __name__ == "__main__":
    main()
    print("Done!")