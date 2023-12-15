
# Notes: Will need to use tensorflow keras functional api as the ABRL layer appears to have two inputs

# Network Architecture
# ====================

# Shallow Feature Extraction, single 3x3 Convolution layer followed by ReLU

# Deep Feature Extraction, 5 layers of 3x3 Convolution layers paired with ReLUs

# Transfer features to HR image space, one convolution layer (Assume from the paper's figure it is 3x3 conv)

# Apply ABRL

# Reshuffle Pixels, labelled as depth to space in paper's figure

# Clip node to restrict pixel values to the range 0 to 255, this gives final output