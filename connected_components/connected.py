import jax
import jax.numpy as jnp

def find_root(parent, i):
    """Find the root of the label and perform path compression."""
    root = i
    while root != parent[root]:
        root = parent[root]
    while i != root:
        j = parent[i]
        parent = jax.ops.index_update(parent, jax.ops.index[i], root)
        i = j
    return root, parent

def union(parent, rank, x, y):
    """Perform union of two sets."""
    root_x, parent = find_root(parent, x)
    root_y, parent = find_root(parent, y)
    if root_x != root_y:
        if rank[root_x] < rank[root_y]:
            parent = jax.ops.index_update(parent, jax.ops.index[root_x], root_y)
        else:
            parent = jax.ops.index_update(parent, jax.ops.index[root_y], root_x)
            if rank[root_x] == rank[root_y]:
                rank = jax.ops.index_add(rank, root_x, 1)
    return parent, rank

def process_pixel(val, xy):
    parent, rank = val
    x, y = xy[0], xy[1]
    current_pixel = image[x, y]
    
    def true_fn(_):
        updates = []
        for dx, dy in [(-1, 0), (0, -1)]:  # Check north and west neighbors
            nx, ny = x + dx, y + dy
            if 0 <= nx < M and 0 <= ny < N and image[nx, ny] == 1:
                index1 = x * N + y
                index2 = nx * N + ny
                parent, rank = union(parent, rank, index1, index2)
                updates.append((parent, rank))
        return updates[-1] if updates else (parent, rank)  # Return the last update or original if no updates
    
    def false_fn(_):
        return parent, rank  # No updates if background
    
    # Execute conditional update based on current pixel being foreground
    parent, rank = jax.lax.cond(current_pixel == 1, true_fn, false_fn, None)
    
    return parent, rank, None


def first_pass(image):
    M, N = image.shape
    parent = jnp.arange(M * N)  # Each pixel is its own parent initially
    rank = jnp.zeros(M * N, dtype=jnp.int32)

    def body_fn(val, xy):
        parent, rank = val
        x, y = xy[0], xy[1]
        current_pixel = image[x, y]
        
        def true_fn(_):
            updates = []
            for dx, dy in [(-1, 0), (0, -1)]:  # Check north and west neighbors
                nx, ny = x + dx, y + dy
                if 0 <= nx < M and 0 <= ny < N and image[nx, ny] == 1:
                    index1 = x * N + y
                    index2 = nx * N + ny
                    parent, rank = union(parent, rank, index1, index2)
                    updates.append((parent, rank))
            return updates[-1] if updates else (parent, rank)  # Return the last update or original if no updates
        
        def false_fn(_):
            return parent, rank  # No updates if background
        
        # Execute conditional update based on current pixel being foreground
        parent, rank = jax.lax.cond(current_pixel == 1, true_fn, false_fn, None)
    
    return parent, rank, None


    indices = jnp.stack(jnp.meshgrid(jnp.arange(M), jnp.arange(N), indexing='ij'), -1).reshape(-1, 2)
    final_parent, final_rank, _ = jax.lax.scan(body_fn, (parent, rank), indices)
    return final_parent, final_rank

def second_pass(image, parent):
    M, N = image.shape
    labels = jnp.zeros(M * N, dtype=jnp.int32)

    def body_fn(val, i):
        labels = val
        root, updated_parent = find_root(parent, i)
        labels = jax.ops.index_update(labels, jax.ops.index[i], root)
        return labels, None

    final_labels, _ = jax.lax.scan(body_fn, labels, jnp.arange(M * N))
    return final_labels.reshape(M, N)

# Example usage
image = jnp.array([[1, 0, 0, 1],
                   [1, 1, 0, 0],
                   [0, 0, 1, 1],
                   [0, 0, 1, 1]])

parent, rank, _ = first_pass(image)
labels = second_pass(image, parent)
print("Final labels:", labels.reshape(image.shape))
