import csv
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import recall_score,precision_score,f1_score

import config

def accuracy_plot(training_results,test_results):
    """ Generates the learning curves.

      Arguments:
        training_results: A tuple with accuracy in the following form:
        (accuracy,loss).
        test_results: A tuple with accuracy in the following form:
        (validation_accuracy,validation_loss).

      Returns:
        None
      """
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(training_results[0], label='Training Accuracy')
    plt.plot(test_results[0], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(training_results[1], label='Training Loss')
    plt.plot(test_results[1], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

def write_to_csv(data,header):
    file_names = data[0]
    class_probabilities = data[1]
    classifications = data[2]
    classifications = ['Yes' if item==1 else 'No' for item in classifications]
    with open("output.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerow(header)
        index = 1
        for name, prob, label in zip(file_names, class_probabilities, classifications):
            csvWriter.writerow([index, name, prob, label])
            index += 1

def plot_confusion_matrix(y_actual,y_predicted):
    cm = confusion_matrix(y_actual,y_predicted)
    ax = plt.subplot()
    sns.heatmap(cm,annot=True,ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    plt.show()

def evaluation(y_actual,y_predicted):
    accuracy = accuracy_score(y_actual,y_predicted)
    recall = recall_score(y_actual,y_predicted)
    precision = precision_score(y_actual,y_predicted)
    f_1 = f1_score(y_actual,y_predicted)
    print(f'Sigmoid Threshold: {config.SIGMOID_THRESHOLD}\n'
          f'Test Accuracy: {accuracy}\n'
          f'Recall Score: {recall}\n'
          f'Precision Score: {precision}\n'
          f'F1 Score: {f_1}\n')