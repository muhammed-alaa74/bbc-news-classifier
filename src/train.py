"""
Train the BBC News classifier end to end.

Run from the project root:
    python -m src.train
"""

import matplotlib
matplotlib.use("Agg")  # no display needed, we just save PNGs
import matplotlib.pyplot as plt

from . import config
from .data_preprocessing import build_dataset, save_artifacts
from .model import build_model


def plot_history(history):
    # Accuracy
    plt.figure()
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Train", "Validation"])
    plt.savefig(f"{config.ASSETS_DIR}/accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Loss
    plt.figure()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train", "Validation"])
    plt.savefig(f"{config.ASSETS_DIR}/loss.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("Loading & preprocessing data...")
    dataset = build_dataset()

    print(f"Classes ({dataset.label_encoder.num_classes}):", dataset.label_encoder.word_index)

    print("Building model...")
    model = build_model(num_classes=dataset.label_encoder.num_classes)
    model.summary()

    print("Training...")
    history = model.fit(
        dataset.train_padded,
        dataset.train_labels,
        epochs=config.NUM_EPOCHS,
        validation_data=(dataset.validation_padded, dataset.validation_labels),
        verbose=2,
    )

    print("Saving model & tokenizers...")
    model.save(config.MODEL_PATH)
    save_artifacts(dataset.tokenizer, dataset.label_encoder)

    print("Saving training curves to assets/...")
    plot_history(history)

    val_loss, val_acc = model.evaluate(dataset.validation_padded, dataset.validation_labels, verbose=0)
    print(f"Final validation accuracy: {val_acc:.4f} | validation loss: {val_loss:.4f}")


if __name__ == "__main__":
    main()
