import os
import joblib
import matplotlib.pyplot as plt


class BaseModel:
    def save(self, history, model_name, base_dir = os.path.abspath("./outputs/")):
        self.model.save(str(base_dir + "/saved_models/" + model_name + ".h5"))
        with open(base_dir + "/train_history/" + str("h_"+ model_name)+".pickle", 'wb') as file:
            joblib.dump(history, file)

    def visualize(self, history, plot_name, path = str(os.path.abspath("./outputs/")) + "/train_history/"):
        print("object to visualize: ", type(history), " > ", history)
        title = plot_name
        fig, axs = plt.subplots(3)
        fig.suptitle(title)
        axs[0].plot(range(0, len(history['loss'])), history['loss'], color = 'red')
        axs[0].set_ylabel('Loss')
        axs[1].plot(range(0, len(history['val_loss'])), history['val_loss'], color = 'blue')
        axs[1].set_ylabel('Validation Loss')
        axs[2].plot(range(0, len(history['mae'])), history['mae'], color = 'green')
        axs[2].set_ylabel('mae')

        plt.legend()
        plt.savefig(path + "/plots/" + str(plot_name) + ".png")


