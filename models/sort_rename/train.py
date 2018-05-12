from model import create_model_bidirectional, create_model_symmetry, create_model_back_forth
from keras.callbacks import TensorBoard, ProgbarLogger, ModelCheckpoint
from batch_generator import BatchGenerator
import getopt, sys

opts, args = getopt.getopt(sys.argv[1:], "s:l:p:w:b:")
opts = dict(opts)
save_to = opts.get("-s")
load_from = opts.get("-l")
workers = int(opts.get("-w", 1))
mbsize = int(opts.get("-b", 10))
if "-p" in opts:
    save_to = opts.get("-p")
    load_from = save_to
    

data_dir = args[0]

nvalidate = 50

dataProvider = BatchGenerator(data_dir, nvalidate = nvalidate)
validate_set = dataProvider.validateSet()
print type(validate_set)

tb = TensorBoard(write_images = False, histogram_freq=1.0)
callbacks = [tb]
if save_to:
    callbacks.append(ModelCheckpoint(filepath=save_to, verbose=1, save_best_only=True, save_weights_only = True, monitor="val_loss"))

model = create_model_symmetry()
#model = create_model_back_forth()

if load_from:
    model.load_weights(load_from)
    print "model weights loaded from %s" % (load_from,)

model.fit_generator(dataProvider.infiniteTrainGenerator(mbsize=mbsize), 
    workers = workers, 
    epochs=100, verbose=1, steps_per_epoch=100, validation_steps = nvalidate,
    callbacks=callbacks,
    validation_data=validate_set)