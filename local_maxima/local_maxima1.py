
import jax.numpy as jnp
from jax import jit
import jax



@jit
def find_local_maxima(image):
    """
    Find local maxima in a 2D array.
    
    Args:
    - image: A 2D array (image) of shape (H, W).
    
    Returns:
    - A boolean array of the same shape as `image`, where True indicates a local maximum.
    """
    # Expand image dimensions to avoid boundary issues
    padded_image = jnp.pad(image, 1, mode='constant', constant_values=jnp.min(image))

    # Shift the image in all eight directions
    shifts = [(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1] if (i, j) != (0, 0)]
    neighbors = [jnp.roll(padded_image, shift=shift, axis=(0, 1)) for shift in shifts]

    # Stack shifted images for comparison
    stacked_neighbors = jnp.stack(neighbors, axis=0)

    # Original image (expanded) repeated for vectorized comparison
    expanded_original = jnp.expand_dims(padded_image, axis=0)

    # True if the original pixel value is greater than all of its neighbors
    local_maxima_mask = jnp.all(expanded_original > stacked_neighbors, axis=0)

    # Remove padding and return the mask indicating local maxima
    return local_maxima_mask[1:-1, 1:-1]


if __name__ == '__main__':
    # # Example usage
    # image_size = (10, 10)
    # # Generate a random image for demonstration
    # image = jax.random.normal(jax.random.PRNGKey(0), (image_size))

    import matplotlib.pyplot as plt
    from skimage import data
    import numpy as np
    import timeit
    from skimage.feature import peak_local_max
    img = jnp.array(data.coins().astype(np.float32)/ 255) 

    # Find local maxima
    local_maxima = find_local_maxima(img)

    # Print result
    plt.imshow(img + local_maxima, cmap='gray')
    plt.title('Image')
    plt.show()

    print('skimage peak_local_max:')
    print(timeit.timeit(lambda: peak_local_max(np.array(img)), number=3))
    print('find_local_maxima compilation:')
    print(timeit.timeit(lambda: find_local_maxima(img), number=1))
    print('find_local_maxima:')
    print(timeit.timeit(lambda: find_local_maxima(img), number=3))
