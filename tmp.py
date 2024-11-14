from bioimageio.core.resource_tests import test_model, debug_model
import matplotlib.pyplot as plt

import instanseg


d1 = debug_model("../models/instanseg/brightfield_nuclei/rdf.yaml")

d2 = debug_model("../models/instanseg/fluorescence_nuclei_and_cells/rdf.yaml")

plt.hist(d1["diff"])


im = instanseg.InstanSeg("brightfield_nuclei")


input = np.load("../models/instanseg/brightfield_nuclei/test-input.npy")
expected = np.load("../models/instanseg/brightfield_nuclei/test-output.npy")


output = im.eval_small_image(input)
instanseg.utils.utils.show_images([input, output[0], expected])


instanseg.utils.utils.show_images([d1["expected"][0], output[0]])
