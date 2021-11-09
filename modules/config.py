import os


IMG_SHAPE = (512, 512, 1)

BATCH_SIZE = 64
EPOCHS = 200

# define the path to the base output directory
BASE_OUTPUT = "./output"

# use the base output path to derive the path to the serialized
# model along with training history plot
MODEL_PATH = os.path.sep.join([BASE_OUTPUT,
	"triplet_loss"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT,
	"triplet_plot.png"])