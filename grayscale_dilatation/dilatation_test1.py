import jax.numpy as jnp
from jax import lax
import jax
from jax import jit
from functools import partial

@partial(jit, static_argnames=('structuring_element_size',))
def jax_grayscale_dilation(image, structuring_element_size):
    """
    Perform grayscale dilation on a 2D image using a vectorized approach.
    For circular structuring element you can apply small window several times.
    For cross you can do line and line....
    """
    # Define a scalar for the initial value that does not depend on the dynamic content of `image`
    # Since we're doing dilation, we can safely set this to the minimum possible value of the image type
    init_value = jnp.finfo(image.dtype).min
    
    # Calculate padding sizes
    pad_size = structuring_element_size // 2
    
    # Pad the image similarly to 'edge' mode in scipy's maximum_filter
    padded_image = jnp.pad(image, pad_width=pad_size, mode='constant', constant_values=init_value)
    
    # Use reduce_window to perform the dilation operation
    dilated_image = lax.reduce_window(padded_image, init_value=init_value,
                                      computation=lambda x, y: jnp.maximum(x, y),
                                      window_dimensions=(structuring_element_size, structuring_element_size),
                                      window_strides=(1, 1), padding='VALID')
    return dilated_image



if __name__ == '__main__':
    # # Example usage
    # image_size = (10, 10)
    # # Generate a random image for demonstration
    # image = jax.random.normal(jax.random.PRNGKey(0), (image_size))

    import matplotlib.pyplot as plt
    from skimage import data
    import numpy as np
    import timeit
    from skimage.morphology import dilation
    img = jnp.array(data.coins().astype(np.float32)/ 255) 

    structuring_element = np.ones((21,21))
    structuring_element_size = structuring_element.shape[0]

    # Find local maxima
    dilated = jax_grayscale_dilation(img, structuring_element_size)
    dilated_skimage = dilation(np.array(img), np.array(structuring_element))

    test =  np.abs(np.array(dilated) - dilated_skimage)

    # Print result
    plt.imshow(img)
    plt.show()
    plt.imshow(dilated)
    plt.show()
    plt.imshow(dilated_skimage)
    plt.show()
    plt.imshow(test)
    plt.show()


    print('skimage dilation:')
    print(timeit.timeit(lambda: dilation(np.array(img), np.array(structuring_element)), number=3))
    print('jax_grayscale_dilation compilation:')
    print(timeit.timeit(lambda: jax_grayscale_dilation(img, structuring_element_size), number=1))
    print('jax_grayscale_dilation:')
    print(timeit.timeit(lambda: jax_grayscale_dilation(img, structuring_element_size), number=3))
