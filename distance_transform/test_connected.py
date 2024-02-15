import jax
import jax.numpy as jnp

def prepare_data(image):
    M, N = image.shape
    # Generate a grid of indices for the image
    i, j = jnp.indices((M, N))
    indices = jnp.stack([i, j], axis=-1)  # Shape: (M, N, 2)
    
    # Define neighbor offsets for 4-connectivity
    neighbors_offset = jnp.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

    # For each pixel, compute indices of its neighbors, taking care to avoid out-of-bound indices
    neighbor_indices = indices[:, :, None, :] + neighbors_offset[None, None, :, :]  # Shape: (M, N, 4, 2)
    neighbor_indices = jnp.clip(neighbor_indices, 0, jnp.array([M-1, N-1]))

    # Flatten the arrays for easier processing with jax.lax.scan
    flat_indices = indices.reshape(-1, 2)
    flat_neighbors_indices = neighbor_indices.reshape(-1, 4, 2)
    
    return flat_indices, flat_neighbors_indices

def process_element(carry, x):
    pixel_index, neighbor_indices = x
    # Placeholder for processing logic, e.g., labeling based on neighbors
    # For demonstration, simply return the carry
    return carry, (pixel_index, neighbor_indices)

def two_pass_algorithm(image):
    flat_indices, flat_neighbors_indices = prepare_data(image)
    initial_carry = 0  # Placeholder for any initial state you might need

    # First pass: process each pixel and its neighbors
    _, processed = jax.lax.scan(process_element, initial_carry, (flat_indices, flat_neighbors_indices))

    # Second pass: could be another jax.lax.scan or different processing based on `processed`
    # Placeholder for the second pass logic

    return processed

image = jnp.array([[1, 0, 0, 1],
                [1, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 1]])
processed = two_pass_algorithm(image)
print("Processed data:", processed)
