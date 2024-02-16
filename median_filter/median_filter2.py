import jax
import jax.numpy as jnp
from jax import jit
from functools import partial


@partial(jit, static_argnames=("kernel_size",))
def median_filter(image, kernel_size):
    # Calculate the number of sliding windows that fit in each dimension
    # taky zere hodně paměti a bypisuje nejakej error, not working!!!!
    num_windows_y = image.shape[0] - kernel_size + 1
    num_windows_x = image.shape[1] - kernel_size + 1
    
    # Prepare the start indices for all windows
    start_indices = [(i, j) for i in range(num_windows_y) for j in range(num_windows_x)]
    
    # A helper function to extract a single window using dynamic_slice
    def extract_median(start_index):
        return jnp.median(jax.lax.dynamic_slice(image, start_index, (kernel_size, kernel_size)))
    
    # Extract all windows using vmap for vectorization
    medians = jax.vmap(extract_median)(jnp.array(start_indices))
    return medians.reshape((num_windows_y, num_windows_x))

# Example usage
if __name__ == "__main__":
    # Create a dummy image
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.io import imread
    from scipy.ndimage import median_filter as median_filter_ndimage
    import timeit

    img = imread("../lena.png")[:,:,0] / 255
    img = img + 0.3 * np.ones_like(img) * (np.random.rand(*img.shape) > 0.9)
    img = jnp.array(img) 
    plt.imshow(img, cmap="gray")
    plt.show()

    kernel_size = 51

    print('skimage peak_local_max:')
    print(timeit.timeit(lambda: median_filter_ndimage(np.array(img), kernel_size), number=3))
    print('median_filter compilation:')
    print(timeit.timeit(lambda: median_filter(img, kernel_size), number=1))
    print('median_filter:')
    print(timeit.timeit(lambda: median_filter(img, kernel_size), number=3))


    filtered_image = median_filter(img, kernel_size)

    plt.imshow(filtered_image, cmap="gray")
    plt.show()

