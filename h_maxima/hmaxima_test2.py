import jax.numpy as jnp
from jax import jit, lax
import jax
from functools import partial


@partial(jit, static_argnames=('iterations',))
def morphological_reconstruction(seed, mask, iterations):
    """Simplified morphological reconstruction using dilation and clipping."""
    current = seed
    for _ in range(iterations):
        dilated = lax.reduce_window(current, 0.0, lax.max,
                                      (3, 3),
                                      (1, 1), 'SAME')
        current = jnp.minimum(dilated, mask)
    return current


@partial(jit, static_argnames=('iterations',))
def h_maxima(image, h, iterations):
    """Identify and retain local maxima that are at least h units higher than their surroundings.
    - iteratons should be sufficint to converge (while loop is slow in jax) (img width - but still dont have to converge)
    - expect image to be in range (0, inf) for correct padding in dilation
    - to meke if faster you can maybe enlarge structuring element, but dont know how it affects the result
    """

    rec_img = morphological_reconstruction(image, image + h, iterations)
    residue_img = image - rec_img + h
    significant_maxima = residue_img >= h
    return significant_maxima.astype(jnp.uint8)

if __name__ == '__main__':
    # image_size = (10, 10)
    # img = jax.random.normal(jax.random.PRNGKey(0), (image_size))

    import matplotlib.pyplot as plt
    from skimage import data
    import numpy as np
    import timeit
    from skimage.morphology import h_maxima as h_maxima_skimage

    
    img = data.coins().astype(np.float32)/ 255
    img = jnp.array(img)

    # Find local maxima
    local_maxima = h_maxima(img, 0.15, img.shape[0])

    # Print result
    plt.imshow(img + local_maxima, cmap='gray')
    plt.title('Image')
    plt.show()

    # Find local maxima
    local_maxima = h_maxima_skimage(img, 0.15)

    # Print result
    plt.imshow(img + local_maxima, cmap='gray')
    plt.title('Image')
    plt.show()


    print('h_maxima:')
    print(timeit.timeit(lambda: h_maxima_skimage(img, 0.15), number=3))
    print('h_maxima compilation:')
    print(timeit.timeit(lambda: h_maxima(img, 0.15, img.shape[0]), number=1))
    print('h_maxima:')
    print(timeit.timeit(lambda: h_maxima(img, 0.15, img.shape[0]), number=3))
