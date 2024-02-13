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
def my_connected_comp(img, indices):
    foreground = img > 0

    max_idx  = reduce(mul, img.shape) + 1
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


    indices = indices * foreground 
    

    return img, indices

max_idx  = reduce(mul, img.shape) + 1
largest_value = max_idx
indices = jnp.arange(1, max_idx).reshape(img.shape)

indices_previdous_current = jnp.stack([jnp.zeros_like(indices), indices], axis=-1)

@jit
def loop_twice(img, indices):
    for _ in range(2):
        img, indices = my_connected_comp(img, indices_previdous_current)
    return img, indices




print('skimage:')
print(timeit.timeit(lambda: label(img > 0), number=3))
print('my_connected_comp_check compilation:')
print(timeit.timeit(lambda: loop_twice(img, indices), number=1))
print('my_connected_comp_check:')
print(timeit.timeit(lambda: loop_twice(img, indices), number=3))
