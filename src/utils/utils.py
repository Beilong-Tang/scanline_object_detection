import numpy as np

class BoxObject:
    def __init__(self, y_start, x_start, y_end, x_end, thickness, cls, line):
        """
        line: numpy array of shape [width, height, C] the width ususally will be 1 or 2
        """
        y_start, x_start, y_end, x_end = int(y_start), int(x_start), int(y_end), int(x_end)
        self.y_start = y_start
        self.x_start = x_start 
        self.x_end = x_end 
        self.y_end = y_end
        self.cls = int(cls)
        self.center = (int((x_start + x_end) / 2), int((y_end + y_start) / 2))
        self.interval = y_end - y_start ## The maximum width
        self.direction = None # Direction , 1 means go up, 0 means go down. None means not specificed
        self.objs = [line[:, y_start:y_start + self.interval]] # This stores the real object



def pad_list(xs, pad_value, mode = 'constant'):
    """Pads a list of NumPy arrays along the second dimension evenly.

    Args:
        xs (List[np.ndarray]): List of arrays [(N_1, T_1, *), (N_2, T_2, *), ..., (N_B, T_B, *)].
        pad_value (float): Value for padding.

    Returns:
        List[np.ndarray]: List of evenly padded arrays with the same second dimension (Tmax).
    """
    max_len = max(x.shape[1] for x in xs)  # Maximum size in the second dimension
    padded_xs = []
    for x in xs:
        pad_total = max_len - x.shape[1]  # Total padding needed
        pad_before = pad_total // 2       # Padding before the original values
        pad_after = pad_total - pad_before  # Remaining padding goes to the end

        pad_width = [(0, 0)] * x.ndim  # No padding for other dimensions
        pad_width[1] = (pad_before, pad_after)  # Pad evenly along the second dimension
        if mode == 'constant':
            padded_x = np.pad(x, pad_width, mode=mode, constant_values=pad_value)
        elif mode == 'edge':
            padded_x = np.pad(x, pad_width, mode=mode)
        padded_xs.append(padded_x)

    return padded_xs
