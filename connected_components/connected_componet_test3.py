import jax
import jax.numpy as jnp
from functools import reduce
from operator import mul
import timeit 
from skimage.measure import label
from jax import jit
import matplotlib.pyplot as plt
from jax import random
from functools import partial



key = random.PRNGKey(0)
num_circles = 10
image_size = (1000, 1000)
radius_range = (50, 150)

img = jnp.zeros(image_size)
height, width = image_size
min_radius, max_radius = radius_range

# Generate random positions and radii for the circles
keys = random.split(key, num=num_circles*3)  # Split the key for x, y, and radius
xs = random.randint(keys[0], (num_circles,), 0, width)
ys = random.randint(keys[1], (num_circles,), 0, height)
radii = random.randint(keys[2], (num_circles,), min_radius, max_radius)

x_grid, y_grid = jnp.meshgrid(jnp.arange(width), jnp.arange(height))

for x, y, radius in zip(xs, ys, radii):
    mask = ((x_grid - x) ** 2 + (y_grid - y) ** 2) <= radius ** 2
    img = img.at[mask].set(1)


plt.imshow(img, cmap='gray')
plt.show()


@jit
def shift_left(image, border_value=0):
    shifted = jnp.roll(image, shift=-1, axis=1)  # Shift left
    shifted = shifted.at[:, -1].set(border_value)  # Fill right border
    return shifted

@jit
def shift_up(image, border_value=0):
    shifted = jnp.roll(image, shift=-1, axis=0)  # Shift up
    shifted = shifted.at[-1, :].set(border_value)  # Fill bottom border
    return shifted

@jit
def shift_right(image, border_value=0):
    shifted = jnp.roll(image, shift=1, axis=1)  # Shift right
    shifted = shifted.at[:, 0].set(border_value)  # Fill left border
    return shifted

@jit
def shift_down(image, border_value=0):
    shifted = jnp.roll(image, shift=1, axis=0)  # Shift down
    shifted = shifted.at[0, :].set(border_value)  # Fill top border
    return shifted


@jit
def my_connected_comp_while_loop(img):
    foreground = img > 0
    max_idx = reduce(mul, img.shape) + 1
    largest_value = max_idx
    indices_previous = jnp.arange(1, max_idx).reshape(img.shape)
    indices_previous = indices_previous * foreground + largest_value * (1 - foreground)

    def body(val):
        indices_previous = val
        all_neighbors = jnp.stack([indices_previous,
                                   shift_left(indices_previous, border_value=largest_value),
                                   shift_up(indices_previous, border_value=largest_value),
                                   shift_right(indices_previous, border_value=largest_value),
                                   shift_down(indices_previous, border_value=largest_value)],
                                  axis=-1)

        indices = jnp.min(all_neighbors, axis=-1)
        indices = indices * foreground + largest_value * (1 - foreground)
        return indices

    def cond(val1, val2):
        return jnp.any(val1 != val2)

    # Initialize loop variables
    initial_val = indices_previous
    # Run the while loop
    result = jax.lax.while_loop(lambda vals: cond(vals[0], vals[1]), 
                            lambda vals: (vals[1], body(vals[1])),
                            (initial_val, body(initial_val)))

    indices = result[0] * foreground
    return indices


@jit
def my_connected_comp(img):
    foreground = img > 0

    max_idx  = reduce(mul, img.shape) + 1
    indices = jnp.arange(1, max_idx).reshape(img.shape)
    largest_value = max_idx


    for iter in range(500):
        indices = indices * foreground + largest_value * (1 - foreground)
        # Define neighbor offsets for 4-connectivity
        all_neighbors = jnp.stack([indices,
                                    shift_left(indices, border_value=largest_value),
                                    shift_up(indices, border_value=largest_value),
                                    shift_right(indices, border_value=largest_value),
                                    shift_down(indices, border_value=largest_value)],
                                    axis=-1)

        indices = jnp.min(all_neighbors, axis=-1)
        if iter == 200:
            break


    indices = indices * foreground 
    

    return indices









@jit
def my_connected_comp_fori(indices, foreground, largest_value):
    foreground = img > 0

    max_idx  = reduce(mul, img.shape) + 1
    indices = jnp.arange(1, max_idx).reshape(img.shape)
    largest_value = max_idx


    def body(i, indices):
        indices = indices * foreground + largest_value * (1 - foreground)
        # Define neighbor offsets for 4-connectivity
        all_neighbors = jnp.stack([indices,
                                    shift_left(indices, border_value=largest_value),
                                    shift_up(indices, border_value=largest_value),
                                    shift_right(indices, border_value=largest_value),
                                    shift_down(indices, border_value=largest_value)],
                                    axis=-1)

        return jnp.min(all_neighbors, axis=-1)

    indices = jax.lax.fori_loop(0, 500, body, indices)

    indices = indices * foreground 

    return indices



print('skimage:')
print(timeit.timeit(lambda: label(img > 0), number=3))
print('for compilation:')
print(timeit.timeit(lambda: my_connected_comp(img), number=1))
print('for:')
print(timeit.timeit(lambda: my_connected_comp(img), number=3))
print('my_connected_comp_fori compilation:')
print(timeit.timeit(lambda: my_connected_comp_fori(img), number=1))
print('my_connected_comp_fori:')
print(timeit.timeit(lambda: my_connected_comp_fori(img), number=3))