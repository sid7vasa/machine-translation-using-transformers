import tensorflow as tf
from utils.dataset import load_datasets
from utils.helper import masked_loss, masked_accuracy, CustomSchedule
from models.transformer import Transformer

if __name__=="__main__":
    import sys
    sys.path.append(r"C:\Users\santo\workspace\machine-translation-using-transformers")
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1
    tokenizers, train_batches, val_batches = load_datasets()

    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                        epsilon=1e-9)

    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
        target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
        dropout_rate=dropout_rate)

    transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy])

    transformer.fit(train_batches,
                epochs=20,
                validation_data=val_batches)

