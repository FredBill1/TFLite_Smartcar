import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class HistoryViewer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.train_acc = []
        self.train_loss = []
        self.val_acc = []
        self.val_loss = []

    def update(self, history):
        self.train_acc += history.history["accuracy"]
        self.train_loss += history.history["loss"]
        self.val_acc += history.history["val_accuracy"]
        self.val_loss += history.history["val_loss"]

    def show(self):
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        plt.plot(self.train_acc, label="Training Accuracy")
        plt.plot(self.val_acc, label="Validation Accuracy")
        plt.legend(loc="lower right")
        plt.ylabel("Accuracy")
        # plt.ylim([min(plt.ylim()),1])
        plt.title("Training and Validation Accuracy")

        plt.subplot(2, 1, 2)
        plt.plot(self.train_loss, label="Training Loss")
        plt.plot(self.val_loss, label="Validation Loss")
        plt.legend(loc="upper right")
        plt.ylabel("Cross Entropy")
        # plt.ylim([0,1.0])
        plt.title("Training and Validation Loss")
        plt.xlabel("epoch")
        plt.show()
