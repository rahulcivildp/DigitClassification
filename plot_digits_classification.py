"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""
import utils

def main():
    digits = utils.load_digits_data()

    utils.display_training_images(digits)

    data = utils.flatten_images(digits)

    clf = utils.create_classifier()

    X_train, X_test, y_train, y_test = utils.split_dataset(
        data,
        digits.target
    )

    trained_clf = utils.train_classifier(clf, X_train, y_train)

    predicted = utils.predict_digits(trained_clf, X_test)

    utils.display_predictions(X_test, predicted)

    utils.print_classification_report(trained_clf, y_test, predicted)

    disp = utils.display_confusion_matrix(y_test, predicted)

    utils.rebuild_classification_report(disp)

    utils.show_plots()


main()
