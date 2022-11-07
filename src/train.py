import yaml
with open('../parameters.yaml', 'r') as file:
    parameters = yaml.safe_load(file)
import sys
sys.path.append(parameters['project_dir'])
import tensorflow as tf
from utils.dataset import load_datasets
from utils.helper import masked_loss, masked_accuracy, CustomSchedule
from models.transformer import Transformer

if __name__=="__main__":
    with open('../parameters.yaml', 'r') as file:
        parameters = yaml.safe_load(file)

    tokenizers, train_batches, val_batches = load_datasets()

    learning_rate = CustomSchedule(parameters['architecture']['d_model'])

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                        epsilon=1e-9)

    transformer = Transformer(
        num_layers=parameters['architecture']['num_layers'],
        d_model=parameters['architecture']['d_model'],
        num_heads=parameters['architecture']['num_heads'],
        dff=parameters['architecture']['dff'],
        input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
        target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
        dropout_rate=parameters['architecture']['dropout_rate'])

    transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy])

    transformer.fit(train_batches,
                epochs=20,
                validation_data=val_batches)

    transformer.save("transformer.h5")

