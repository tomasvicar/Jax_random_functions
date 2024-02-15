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


# # @jit
# def horizontal_vertical_pass(image):
#     # Assuming 'image' is a binary image with foreground=1, background=0
#     # Invert image: foreground=0 (for distance computation), background=Inf
#     # inverted_image = jnp.where(image == 1, 0, jnp.inf)
    
#     # Horizontal pass
#     left_pass = jnp.fliplr(jnp.cumsum(jnp.fliplr(inverted_image), axis=1))
#     right_pass = jnp.cumsum(inverted_image, axis=1)
#     horizontal_dist = jnp.minimum(left_pass, right_pass)
    
#     # Vertical pass
#     up_pass = jnp.flipud(jnp.cumsum(jnp.flipud(inverted_image), axis=0))
#     down_pass = jnp.cumsum(inverted_image, axis=0)
#     vertical_dist = jnp.minimum(up_pass, down_pass)

#     plt.imshow(left_pass, cmap='gray')
#     plt.show()
#     plt.imshow(right_pass, cmap='gray')
#     plt.show()
#     plt.imshow(up_pass, cmap='gray')
#     plt.show()
#     plt.imshow(down_pass, cmap='gray')
#     plt.show()

    
#     # Combine horizontal and vertical passes
#     combined_dist = jnp.minimum(horizontal_dist, vertical_dist)
    
#     return combined_dist



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


plt.imshow(np.cumsum(img), cmap='gray')
plt.show()



# print('horizontal_vertical_pass compilation:')
# print(timeit.timeit(lambda: horizontal_vertical_pass(img), number=1))
# print('horizontal_vertical_pass:')
# print(timeit.timeit(lambda: horizontal_vertical_pass(img), number=3))


plt.imshow(horizontal_vertical_pass(img), cmap='gray')
plt.show()
