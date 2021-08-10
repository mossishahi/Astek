import os
import joblib
import matplotlib.pyplot as plt

class BaseModule:
    def save(self, history, module_name, base_dir = os.path.abspath("./outputs/")):
        self.model.save(str(base_dir + "/saved_models/" + module_name + ".h5"))
        with open(base_dir + "/train_history/" + str("h_"+ module_name)+".pickle", 'wb') as file:
            joblib.dump(history.history, file)
    def visualize(self, history, plot_name, path = str(os.path.abspath("./outputs/")) + "/train_history/"):
        print("object to visualize: ", type(history), " > ", history)
        title = plot_name
        fig, axs = plt.subplots(2)
        fig.suptitle(title)
        axs[0].plot(history['loss'])
        axs[0].set_ylabel('Loss')
        axs[1].plot(history['val_loss'])
        axs[1].set_ylabel('Validation Loss')
        fig = plt.figure(figsize=(16,10))
        plt.show()
        plt.savefig(path + "/plots/" + str(plot_name) + ".png")

