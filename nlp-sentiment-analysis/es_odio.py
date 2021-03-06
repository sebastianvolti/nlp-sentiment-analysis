import sys
from constants import MODEL_TYPES
import data
import model

def main():
    path = sys.argv[1]
    total_files = len(sys.argv) - 2
    test_names = []

    for i in range(total_files):
        test_names.append(sys.argv[i + 2])

    datasets = data.Data(path, test_names)
    model_type = MODEL_TYPES["CONVOLUTIONAL"]  # HERE GOES THE MODEL WE CONFIGURED IN CORSS VALIDATION
    epochs = 10                        # HERE GOES THE EPOCHS WE CONFIGURED IN CORSS VALIDATION
    neurons = 64                      # HERE GOES THE NEURONS WE CONFIGURED IN CORSS VALIDATION
    dropout = 0.1                      # HERE GOES THE DROPOUT WE CONFIGURED IN CORSS VALIDATION
    batches_size = 64                 # HERE GOES THE BATCH_SIZE WE CONFIGURED IN CORSS VALIDATION
    m = model.Model(
        model_type=model_type,
        train_dataset=datasets.train,
        neurons=neurons,
        dropout=dropout,
        val_dataset=datasets.test,
        test_files=datasets.test_files,
        path=path
    )
    m.train(epochs, batches_size)
    m.eval()

if __name__ == "__main__":
    main()