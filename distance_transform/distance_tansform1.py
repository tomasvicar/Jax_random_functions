import jax.numpy as jnp
from jax import jit

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


# Function to shift the image in four directions and find the minimum label
@jit
def propagate_min_labels(img):
    # Shift the image in all four directions
    up = jnp.roll(img, shift=-1, axis=0)
    down = jnp.roll(img, shift=1, axis=0)
    left = jnp.roll(img, shift=-1, axis=1)
    right = jnp.roll(img, shift=1, axis=1)

    # Take the minimum across all shifted images and the original
    # This effectively propagates the minimum label to each pixel from its neighbors
    return jnp.minimum(jnp.minimum(jnp.minimum(up, down), left), right)

# Function to iteratively apply the propagate_min_labels function until convergence
@jit
def iterative_min_label_propagation(img, iterations=500):
    for _ in range(iterations):
        img = propagate_min_labels(img)
    return img



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



print('skimage:')
print(timeit.timeit(lambda: label(img > 0), number=3))
print('iterative_min_label_propagation compilation:')
print(timeit.timeit(lambda: iterative_min_label_propagation(img), number=1))
print('iterative_min_label_propagation:')
print(timeit.timeit(lambda: iterative_min_label_propagation(img), number=3))


plt.imshow(iterative_min_label_propagation(img), cmap='gray')
plt.show()
