import jax
import jax.numpy as jnp
from functools import reduce
from operator import mul


# Example usage
img = jnp.array([[1, 0, 0, 1],
                [1, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 1]])


import jax.numpy as jnp

def shift_left(image, border_value=0):
    shifted = jnp.roll(image, shift=-1, axis=1)  # Shift left
    shifted = shifted.at[:, -1].set(border_value)  # Fill right border
    return shifted

def shift_up(image, border_value=0):
    shifted = jnp.roll(image, shift=-1, axis=0)  # Shift up
    shifted = shifted.at[-1, :].set(border_value)  # Fill bottom border
    return shifted

def shift_right(image, border_value=0):
    shifted = jnp.roll(image, shift=1, axis=1)  # Shift right
    shifted = shifted.at[:, 0].set(border_value)  # Fill left border
    return shifted

def shift_down(image, border_value=0):
    shifted = jnp.roll(image, shift=1, axis=0)  # Shift down
    shifted = shifted.at[0, :].set(border_value)  # Fill top border
    return shifted


foreground = img > 0

indices = jnp.arange(reduce(mul, img.shape)).reshape(img.shape)
largest_value = reduce(mul, img.shape)
indices = indices * foreground + largest_value * (1 - foreground)

for iter in range(10):

    # Define neighbor offsets for 4-connectivity
    all_neighbors = jnp.stack([indices,
                                shift_left(indices, border_value=largest_value),
                                shift_up(indices, border_value=largest_value),
                                shift_right(indices, border_value=largest_value),
                                shift_down(indices, border_value=largest_value)],
                                axis=-1)

    indices = jnp.min(all_neighbors, axis=-1)


print(indices)

