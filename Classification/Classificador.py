import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def normal_data(data):
    import statistics as st
    import numpy as np
    size = len(data[0, :]) - 1

    # Select the label
    label = data[:, size]

    # Removing the ID
    mask = np.arange(1, size)
    data = data[:, mask]

    # Select only the 8 best variables
    if len(data[0, :]) > 8:
        mask = np.arange(8)
        data = data[:, mask]

    # Standardization of the data mean=0 variance=1
    size = len(data[0, :])
    for i in range(0, size):
        sd = st.stdev(data[:, i])
        mean = st.mean(data[:, i])
        data[:, i] = (data[:, i] - mean) / sd

    return [data, label]


def classifier_cross_val(path, classifier):
    import numpy as np
    from sklearn import svm
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from random import randint
    from sklearn.metrics import roc_curve, auc

    # Loading the data
    data = np.genfromtxt(path, delimiter=';', skip_header=1)

    # Normalizing, Selecting Variables and extrating the labels
    data, label = normal_data(data)

    # Balancing the classifier
    cn = data[label == 0, :]
    label_cn = label[(label == 0)]
    ad = data[label == 1, :]
    label_ad = label[(label == 1)]

    # Selecting 80% of the cn cases
    size_cn = len(cn)
    cn_train_indice = np.random.choice(size_cn, int(round((0.8 * size_cn))), replace=False)
    cn_train_indice = np.sort(cn_train_indice)
    cn_train = cn[cn_train_indice, :]
    cn_test = np.delete(cn, cn_train_indice, axis=0)
    cn_train_label = label_cn[cn_train_indice]
    cn_test_label = np.delete(label_cn, cn_train_indice, axis=0)

    # Selecting 80% of the ad cases
    size_ad = len(ad)
    ad_train_indice = np.random.choice(size_ad, int(round((0.8 * size_ad))), replace=False)
    ad_train_indice = np.sort(ad_train_indice)
    ad_train = ad[ad_train_indice, :]
    ad_test = np.delete(ad, ad_train_indice, axis=0)
    ad_train_label = label_ad[ad_train_indice]
    ad_test_label = np.delete(label_ad, ad_train_indice, axis=0)

    # Constructing the train and test data sets
    train = np.concatenate([ad_train, cn_train])
    train_label = np.concatenate([ad_train_label, cn_train_label])
    test = np.concatenate([ad_test, cn_test])
    test_label = np.concatenate([ad_test_label, cn_test_label])

    # Building the model
    if classifier == 'svm':
        clf = svm.SVC(gamma='scale', kernel='rbf', probability=True)

    elif classifier == 'adb':
        svc = svm.SVC(gamma='scale', kernel='linear', probability=True)
        clf = AdaBoostClassifier(n_estimators=50, learning_rate=1, base_estimator=svc)

    elif classifier == 'lgr':
        clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto')

    # Cross validation the method
    cv = 8
    indice = []

    # Assuring the balance at each fold
    indice_rad = np.random.choice(len(ad_train_label), len(ad_train_label), replace=False)
    indice_rcn = np.random.choice(len(cn_train_label), len(cn_train_label), replace=False)
    indice_rcn = indice_rcn + len(ad_train_label)
    size_indice = len(train_label)
    check = int(round(size_indice / cv))

    k = 0
    for i in range(0, cv):
        vector = []
        if (len(cn_train_indice) - len(ad_train_indice)) == 1:
            for j in range(k, (k + int(check / 2))):
                vector.append(indice_rad[j])
                vector.append(indice_rcn[j])
                k = k + 1
            if k == 16:
                vector.append(indice_rcn[k])
        elif (len(cn_train_indice) - len(ad_train_indice)) == 2:
            if i != (cv - 1):
                for j in range(k, (k + int(check / 2))):
                    vector.append(indice_rad[j])
                    vector.append(indice_rcn[j])
                    k = k + 1
            else:
                a1 = randint(0, (len(ad_train_indice) - 1))
                vector.append(indice_rad[a1])
                a2 = randint(0, (len(ad_train_indice) - 1))
                vector.append(indice_rad[a2])
                vector.append(indice_rcn[k])
                vector.append(indice_rcn[k + 1])
        elif (len(cn_train_indice) - len(ad_train_indice)) == 3:
            if i != (cv - 1):
                for j in range(k, (k + int(check / 2))):
                    vector.append(indice_rad[j])
                    vector.append(indice_rcn[j])
                    k = k + 1
            else:
                a1 = randint(0, (len(ad_train_indice) - 1))
                vector.append(indice_rad[a1])
                a2 = randint(0, (len(ad_train_indice) - 1))
                vector.append(indice_rad[a2])
                a3 = randint(0, (len(ad_train_indice) - 1))
                vector.append(indice_rad[a3])
                vector.append(indice_rcn[k])
                vector.append(indice_rcn[k + 1])
                vector.append(indice_rcn[k + 2])

        indice.append(vector)
    size_cv = len(indice)
    models_cv = []
    accuracy_cv = []
    sensitivity_cv = []
    specificity_cv = []

    for i in range(0, size_cv):
        test_cv = train[indice[i], :]
        test_cv_l = train_label[indice[i]]
        train_cv = np.delete(train, indice[i], axis=0)
        train_cv_l = np.delete(train_label, indice[i], axis=0)
        clf.fit(train_cv, train_cv_l)
        model = clf
        models_cv.append(model)
        prediction = clf.predict(test_cv)
        tn, fp, fn, tp = confusion_matrix(test_cv_l, prediction).ravel()
        accuracy = (tn + tp) / (tn + fp + fn + tp)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        accuracy_cv.append(accuracy)
        sensitivity_cv.append(sensitivity)
        specificity_cv.append(specificity)

    r = 0
    indice_best = 0
    for i in range(0, len(models_cv)):
        if accuracy_cv[i] > r:
            r = accuracy_cv[i]
            indice_best = i

    # Selecting the best model
    model_best = models_cv[indice_best]

    # Prediction and evaluation
    prediction = model_best.predict(test)
    prob = model_best.predict_proba(test)

    # Confusion Matrix evaluation
    tn, fp, fn, tp = confusion_matrix(test_label, prediction).ravel()
    accuracy = (tn + tp) / (tn + fp + fn + tp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    metrics = np.array([accuracy, sensitivity, specificity])

    # ROC curve
    fpr, tpr, thresholds = roc_curve(test_label, prob[:, 1])
    roc_auc = auc(fpr, tpr)
    roc = [fpr, tpr, roc_auc]

    return [model_best, metrics, roc]


def avaliar(path, classifier='svm', number=2000):
    import statistics as st
    import numpy as np
    from scipy import interp
    from sklearn.metrics import auc
    accuracy = []
    sensitivity = []
    specificity = []
    models = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for i in range(0, number):
        aval = classifier_cross_val(path, classifier)
        metrics = aval[1]
        models.append(aval[0])
        accuracy.append(metrics[0])
        sensitivity.append(metrics[1])
        specificity.append(metrics[2])

        roccurve = aval[2]
        fpr = roccurve[0]
        tpr = roccurve[1]
        aucv = roccurve[2]
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        aucs.append(aucv)

    r = 0
    indice = 0
    count = 0
    for i in range(0, len(accuracy)):
        if accuracy[i] > r:
            r = accuracy[i]
            indice = i

        if accuracy[i] == 1:
            count = count + 1

    model_final = models[indice]
    mean_accuracy = st.mean(accuracy) * 100
    mean_sensitivity = st.mean(sensitivity) * 100
    mean_specificity = st.mean(specificity) * 100
    low_classifiers = np.percentile(accuracy, 10) * 100
    count = (count / number) * 100
    results = [mean_accuracy, mean_sensitivity, mean_specificity, low_classifiers, count]

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    graphic = [mean_fpr, mean_tpr, mean_auc, std_auc]
    return [model_final, results, graphic]


def evaluation(classifier, number, fs, plot=False):
    # Structural
    print('Structural classifiers\n')
    os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\MRI\\Estrutural\\Cobra\\GM')
    path_1 = fs
    model, results, graphic = avaliar(path_1, classifier, number)
    print('Cobra GM model accuracy: ', results[0], '%')
    print('Cobra GM model sensitivity: ', results[1], '%')
    print('Cobra GM model specificity: ', results[2], '%')
    print('90% of classifiers perform above: ', results[3], '%')
    print('Percentage of full accuracy: ', results[4], '%')
    if plot is True:
        x = graphic[0]
        y = graphic[1]
        auc = graphic[2]
        sd_auc = graphic[3]
        plt.figure(num=None, figsize=(8,6))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)
        plt.plot(x, y, 'b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (auc, sd_auc), lw=2, alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Cobra GM model mean ROC curve')
        plt.legend(loc="lower right")
        plt.savefig('D:\\Universidade\\5\\Tese\\Escrita\\ROC curves\\Cobra_GM.png', format='png')

    print('\n')

    os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\MRI\\Estrutural\\Cobra\\WM')
    path_1 = fs
    model, results, graphic = avaliar(path_1, classifier, number)
    print('Cobra WM model accuracy: ', results[0], '%')
    print('Cobra WM model sensitivity: ', results[1], '%')
    print('Cobra WM model specificity: ', results[2], '%')
    print('90% of classifiers perform above: ', results[3], '%')
    print('Percentage of full accuracy: ', results[4], '%')
    if plot is True:
        x = graphic[0]
        y = graphic[1]
        auc = graphic[2]
        sd_auc = graphic[3]
        plt.figure(num=None, figsize=(8,6))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)
        plt.plot(x, y, 'b', label=r'Mean ROC (AUC = %0.2f +/- %0.2f)' % (auc, sd_auc), lw=2, alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Cobra WM model mean ROC curve')
        plt.legend(loc="lower right")
        plt.savefig('D:\\Universidade\\5\\Tese\\Escrita\\ROC curves\\Cobra_WM.png', format='png')

    print('\n')

    os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\MRI\\Estrutural\\Hammers\\GM')
    path_1 = fs
    model, results, graphic = avaliar(path_1, classifier, number)
    print('Hammers GM model accuracy: ', results[0], '%')
    print('Hammers GM model sensitivity: ', results[1], '%')
    print('Hammers GM model specificity: ', results[2], '%')
    print('90% of classifiers perform above: ', results[3], '%')
    print('Percentage of full accuracy: ', results[4], '%')
    if plot is True:
        x = graphic[0]
        y = graphic[1]
        auc = graphic[2]
        sd_auc = graphic[3]
        plt.figure(num=None, figsize=(8,6))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)
        plt.plot(x, y, 'b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (auc, sd_auc), lw=2, alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Hammers GM model mean ROC curve')
        plt.legend(loc="lower right")
        plt.savefig('D:\\Universidade\\5\\Tese\\Escrita\\ROC curves\\Hammers_GM.png', format='png')

    print('\n')

    os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\MRI\\Estrutural\\Hammers\\WM')
    path_1 = fs
    model, results, graphic = avaliar(path_1, classifier, number)
    print('Hammers WM model accuracy: ', results[0], '%')
    print('Hammers WM model sensitivity: ', results[1], '%')
    print('Hammers WM model specificity: ', results[2], '%')
    print('90% of classifiers perform above: ', results[3], '%')
    print('Percentage of full accuracy: ', results[4], '%')
    if plot is True:
        x = graphic[0]
        y = graphic[1]
        auc = graphic[2]
        sd_auc = graphic[3]
        plt.figure(num=None, figsize=(8,6))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)
        plt.plot(x, y, 'b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (auc, sd_auc), lw=2, alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Hammers WM model mean ROC curve')
        plt.legend(loc="lower right")
        plt.savefig('D:\\Universidade\\5\\Tese\\Escrita\\ROC curves\\Hammers_WM.png', format='png')

    print('\n')

    os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\MRI\\Estrutural\\Hammers\\CSF')
    path_1 = fs
    model, results, graphic = avaliar(path_1, classifier, number)
    print('Hammers CSF model accuracy: ', results[0], '%')
    print('Hammers CSF model sensitivity: ', results[1], '%')
    print('hammers CSF model specificity: ', results[2], '%')
    print('90% of classifiers perform above: ', results[3], '%')
    print('Percentage of full accuracy: ', results[4], '%')
    if plot is True:
        x = graphic[0]
        y = graphic[1]
        auc = graphic[2]
        sd_auc = graphic[3]
        plt.figure(num=None, figsize=(8,6))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)
        plt.plot(x, y, 'b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (auc, sd_auc), lw=2, alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Hammers CSF model mean ROC curve')
        plt.legend(loc="lower right")
        plt.savefig('D:\\Universidade\\5\\Tese\\Escrita\\ROC curves\\Hammers_CSF.png', format='png')

    print('\n')

    os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\MRI\\Estrutural\\lpba40\\GM')
    path_1 = fs
    model, results, graphic = avaliar(path_1, classifier, number)
    print('Lpba40 GM model accuracy: ', results[0], '%')
    print('Lpba40 GM model sensitivity: ', results[1], '%')
    print('Lpba40 GM model specificity: ', results[2], '%')
    print('90% of classifiers perform above: ', results[3], '%')
    print('Percentage of full accuracy: ', results[4], '%')
    if plot is True:
        x = graphic[0]
        y = graphic[1]
        auc = graphic[2]
        sd_auc = graphic[3]
        plt.figure(num=None, figsize=(8,6))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)
        plt.plot(x, y, 'b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (auc, sd_auc), lw=2, alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Lpba40 GM model mean ROC curve')
        plt.legend(loc="lower right")
        plt.savefig('D:\\Universidade\\5\\Tese\\Escrita\\ROC curves\\Lpba40_GM.png', format='png')
    print('\n')

    os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\MRI\\Estrutural\\Neuro\\GM')
    path_1 = fs
    model, results, graphic = avaliar(path_1, classifier, number)
    print('Neuromorphometrics GM model accuracy: ', results[0], '%')
    print('Neuromorphometrics GM model sensitivity: ', results[1], '%')
    print('Neuromorphometrics GM model specificity: ', results[2], '%')
    print('90% of classifiers perform above: ', results[3], '%')
    print('Percentage of full accuracy: ', results[4], '%')
    if plot is True:
        x = graphic[0]
        y = graphic[1]
        auc = graphic[2]
        sd_auc = graphic[3]
        plt.figure(num=None, figsize=(8,6))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)
        plt.plot(x, y, 'b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (auc, sd_auc), lw=2, alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Neuromorphometrics GM model mean ROC curve')
        plt.legend(loc="lower right")
        plt.savefig('D:\\Universidade\\5\\Tese\\Escrita\\ROC curves\\Neuto_GM.png', format='png')

    print('\n')

    os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\MRI\\Estrutural\\Neuro\\CSF')
    path_1 = fs
    model, results, graphic = avaliar(path_1, classifier, number)
    print('Neuromorphometrics CSF model accuracy: ', results[0], '%')
    print('Neuromorphometrics CSF model sensitivity: ', results[1], '%')
    print('Neuromorphometrics CSF model specificity: ', results[2], '%')
    print('90% of classifiers perform above: ', results[3], '%')
    print('Percentage of full accuracy: ', results[4], '%')
    if plot is True:
        x = graphic[0]
        y = graphic[1]
        auc = graphic[2]
        sd_auc = graphic[3]
        plt.figure(num=None, figsize=(8,6))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)
        plt.plot(x, y, 'b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (auc, sd_auc), lw=2, alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Neuromorphometrics CSF model mean ROC curve')
        plt.legend(loc="lower right")
        plt.savefig('D:\\Universidade\\5\\Tese\\Escrita\\ROC curves\\Neuto_CSF.png', format='png')

    print('\n')

    # Surface
    print('\nSurface classifiers\n')
    os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\MRI\\Superficie\\a2009\\Gyrification')
    path_1 = fs
    model, results, graphic = avaliar(path_1, classifier, number)
    print('a2009 Gyrification model accuracy: ', results[0], '%')
    print('a2009 Gyrification model sensitivity: ', results[1], '%')
    print('a2009 Gyrification model specificity: ', results[2], '%')
    print('90% of classifiers perform above: ', results[3], '%')
    print('Percentage of full accuracy: ', results[4], '%')
    if plot is True:
        x = graphic[0]
        y = graphic[1]
        auc = graphic[2]
        sd_auc = graphic[3]
        plt.figure(num=None, figsize=(8,6))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)
        plt.plot(x, y, 'b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (auc, sd_auc), lw=2, alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('a2009 Gyrification model mean ROC curve')
        plt.legend(loc="lower right")
        plt.savefig('D:\\Universidade\\5\\Tese\\Escrita\\ROC curves\\a2009_G.png', format='png')

    print('\n')

    os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\MRI\\Superficie\\a2009\\Thickness')
    path_1 = fs
    model, results, graphic = avaliar(path_1, classifier, number)
    print('a2009 Thickness model accuracy: ', results[0], '%')
    print('a2009 Thickness model sensitivity: ', results[1], '%')
    print('a2009 Thickness model specificity: ', results[2], '%')
    print('90% of classifiers perform above: ', results[3], '%')
    print('Percentage of full accuracy: ', results[4], '%')
    if plot is True:
        x = graphic[0]
        y = graphic[1]
        auc = graphic[2]
        sd_auc = graphic[3]
        plt.figure(num=None, figsize=(8,6))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)
        plt.plot(x, y, 'b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (auc, sd_auc), lw=2, alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('a2009 Thickness model mean ROC curve')
        plt.legend(loc="lower right")
        plt.savefig('D:\\Universidade\\5\\Tese\\Escrita\\ROC curves\\a2009_T.png', format='png')

    print('\n')

    os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\MRI\\Superficie\\DK40\\Gyrification')
    path_1 = fs
    model, results, graphic = avaliar(path_1, classifier, number)
    print('DK40 Gyrification model accuracy: ', results[0], '%')
    print('DK40 Gyrification model sensitivity: ', results[1], '%')
    print('DK40 Gyrification model specificity: ', results[2], '%')
    print('90% of classifiers perform above: ', results[3], '%')
    print('Percentage of full accuracy: ', results[4], '%')
    if plot is True:
        x = graphic[0]
        y = graphic[1]
        auc = graphic[2]
        sd_auc = graphic[3]
        plt.figure(num=None, figsize=(8,6))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)
        plt.plot(x, y, 'b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (auc, sd_auc), lw=2, alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('DK40 Gyrification model mean ROC curve')
        plt.legend(loc="lower right")
        plt.savefig('D:\\Universidade\\5\\Tese\\Escrita\\ROC curves\\DK40_G.png', format='png')

    print('\n')

    os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\MRI\\Superficie\\DK40\\Thickness')
    path_1 = fs
    model, results, graphic = avaliar(path_1, classifier, number)
    print('DK40 Thickness model accuracy: ', results[0], '%')
    print('DK40 Thickness model sensitivity: ', results[1], '%')
    print('DK40 Thickness model specificity: ', results[2], '%')
    print('90% of classifiers perform above: ', results[3], '%')
    print('Percentage of full accuracy: ', results[4], '%')
    if plot is True:
        x = graphic[0]
        y = graphic[1]
        auc = graphic[2]
        sd_auc = graphic[3]
        plt.figure(num=None, figsize=(8,6))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)
        plt.plot(x, y, 'b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (auc, sd_auc), lw=2, alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('DK40 Thickness model mean ROC curve')
        plt.legend(loc="lower right")
        plt.savefig('D:\\Universidade\\5\\Tese\\Escrita\\ROC curves\\DK40_T.png', format='png')

    print('\n')

    os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\MRI\\Superficie\\HCP\\Gyrification')
    path_1 = fs
    model, results, graphic = avaliar(path_1, classifier, number)
    print('HCP Gyrification model accuracy: ', results[0], '%')
    print('HCP Gyrification model sensitivity: ', results[1], '%')
    print('HCP Gyrification model specificity: ', results[2], '%')
    print('90% of classifiers perform above: ', results[3], '%')
    print('Percentage of full accuracy: ', results[4], '%')
    if plot is True:
        x = graphic[0]
        y = graphic[1]
        auc = graphic[2]
        sd_auc = graphic[3]
        plt.figure(num=None, figsize=(8,6))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)
        plt.plot(x, y, 'b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (auc, sd_auc), lw=2, alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('HCP Gyrification model mean ROC curve')
        plt.legend(loc="lower right")
        plt.savefig('D:\\Universidade\\5\\Tese\\Escrita\\ROC curves\\HCP_G.png', format='png')

    print('\n')

    os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\MRI\\Superficie\\HCP\\Thickness')
    path_1 = fs
    model, results, graphic = avaliar(path_1, classifier, number)
    print('HCP Thickness model accuracy: ', results[0], '%')
    print('HCP Thickness model sensitivity: ', results[1], '%')
    print('HCP Thickness model specificity: ', results[2], '%')
    print('90% of classifiers perform above: ', results[3], '%')
    print('Percentage of full accuracy: ', results[4], '%')
    if plot is True:
        x = graphic[0]
        y = graphic[1]
        auc = graphic[2]
        sd_auc = graphic[3]
        plt.figure(num=None, figsize=(8,6))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)
        plt.plot(x, y, 'b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (auc, sd_auc), lw=2, alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('HCP Thickness model mean ROC curve')
        plt.legend(loc="lower right")
        plt.savefig('D:\\Universidade\\5\\Tese\\Escrita\\ROC curves\\HCP_T.png', format='png')

    print('\n')

    print('DTI')
    os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\DTI\\Data\\lpba40\\FA')
    path_1 = fs
    model, results, graphic = avaliar(path_1, classifier, number)
    print('Lpba40 FA model accuracy: ', results[0], '%')
    print('Lpba40 FA model sensitivity: ', results[1], '%')
    print('Lpba40 FA model specificity: ', results[2], '%')
    print('90% of classifiers perform above: ', results[3], '%')
    print('Percentage of full accuracy: ', results[4], '%')
    if plot is True:
        x = graphic[0]
        y = graphic[1]
        auc = graphic[2]
        sd_auc = graphic[3]
        plt.figure(num=None, figsize=(8,6))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)
        plt.plot(x, y, 'b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (auc, sd_auc), lw=2, alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Lpba40 FA model mean ROC curve')
        plt.legend(loc="lower right")
        plt.savefig('D:\\Universidade\\5\\Tese\\Escrita\\ROC curves\\Lpba40_FA.png', format='png')

    print('\n')

    os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\DTI\\Data\\lpba40\\MD')
    path_1 = fs
    model, results, graphic = avaliar(path_1, classifier, number)
    print('Lpba MD model accuracy: ', results[0], '%')
    print('Lpba40 MD model sensitivity: ', results[1], '%')
    print('Lpba40 MD model specificity: ', results[2], '%')
    print('90% of classifiers perform above: ', results[3], '%')
    print('Percentage of full accuracy: ', results[4], '%')
    if plot is True:
        x = graphic[0]
        y = graphic[1]
        auc = graphic[2]
        sd_auc = graphic[3]
        plt.figure(num=None, figsize=(8,6))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)
        plt.plot(x, y, 'b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (auc, sd_auc), lw=2, alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Lpba40 MD model mean ROC curve')
        plt.legend(loc="lower right")
        plt.savefig('D:\\Universidade\\5\\Tese\\Escrita\\ROC curves\\Lpba40_MD.png', format='png')

    print('\n')

    os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\DTI\\Data\\Desikan\\FA')
    path_1 = fs
    model, results, graphic = avaliar(path_1, classifier, number)
    print('Desikan FA model accuracy: ', results[0], '%')
    print('Desikan FA model sensitivity: ', results[1], '%')
    print('Desikan FA model specificity: ', results[2], '%')
    print('90% of classifiers perform above: ', results[3], '%')
    print('Percentage of full accuracy: ', results[4], '%')
    if plot is True:
        x = graphic[0]
        y = graphic[1]
        auc = graphic[2]
        sd_auc = graphic[3]
        plt.figure(num=None, figsize=(8,6))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)
        plt.plot(x, y, 'b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (auc, sd_auc), lw=2, alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Desikan FA model mean ROC curve')
        plt.legend(loc="lower right")
        plt.savefig('D:\\Universidade\\5\\Tese\\Escrita\\ROC curves\\Desikan_FA.png', format='png')

    print('\n')

    os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\DTI\\Data\\Desikan\\MD')
    path_1 = fs
    model, results, graphic = avaliar(path_1, classifier, number)
    print('Desikan MD model accuracy: ', results[0], '%')
    print('Desikan MD model sensitivity: ', results[1], '%')
    print('Desikan MD model specificity: ', results[2], '%')
    print('90% of classifiers perform above: ', results[3], '%')
    print('Percentage of full accuracy: ', results[4], '%')
    if plot is True:
        x = graphic[0]
        y = graphic[1]
        auc = graphic[2]
        sd_auc = graphic[3]
        plt.figure(num=None, figsize=(8,6))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)
        plt.plot(x, y, 'b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (auc, sd_auc), lw=2, alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Desikan MD model mean ROC curve')
        plt.legend(loc="lower right")
        plt.savefig('D:\\Universidade\\5\\Tese\\Escrita\\ROC curves\\Desikan_MD.png', format='png')

    print('\n')

    os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\DTI\\Data\\Destrieux\\FA')
    path_1 = fs
    model, results, graphic = avaliar(path_1, classifier, number)
    print('Destrieux FA model accuracy: ', results[0], '%')
    print('Destrieux FA model sensitivity: ', results[1], '%')
    print('Destrieux FA model specificity: ', results[2], '%')
    print('90% of classifiers perform above: ', results[3], '%')
    print('Percentage of full accuracy: ', results[4], '%')
    if plot is True:
        x = graphic[0]
        y = graphic[1]
        auc = graphic[2]
        sd_auc = graphic[3]
        plt.figure(num=None, figsize=(8,6))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)
        plt.plot(x, y, 'b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (auc, sd_auc), lw=2, alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Destrieux FA model mean ROC curve')
        plt.legend(loc="lower right")
        plt.savefig('D:\\Universidade\\5\\Tese\\Escrita\\ROC curves\\Destrieux_FA.png', format='png')

    print('\n')

    os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\DTI\\Data\\Destrieux\\MD')
    path_1 = fs
    model, results, graphic = avaliar(path_1, classifier, number)
    print('Destrieux MD model accuracy: ', results[0], '%')
    print('Destrieux MD model sensitivity: ', results[1], '%')
    print('Destrieux MD model specificity: ', results[2], '%')
    print('90% of classifiers perform above: ', results[3], '%')
    print('Percentage of full accuracy: ', results[4], '%')
    if plot is True:
        x = graphic[0]
        y = graphic[1]
        auc = graphic[2]
        sd_auc = graphic[3]
        plt.figure(num=None, figsize=(8,6))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)
        plt.plot(x, y, 'b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (auc, sd_auc), lw=2, alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Destrieux MD model mean ROC curve')
        plt.legend(loc="lower right")
        plt.savefig('D:\\Universidade\\5\\Tese\\Escrita\\ROC curves\\Destrieux_MD.png', format='png')

    print('\n')

    os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\DTI\\Data\\Hammers\\FA')
    path_1 = fs
    model, results, graphic = avaliar(path_1, classifier, number)
    print('Hammers FA model accuracy: ', results[0], '%')
    print('Hammers FA model sensitivity: ', results[1], '%')
    print('Hammers FA model specificity: ', results[2], '%')
    print('90% of classifiers perform above: ', results[3], '%')
    print('Percentage of full accuracy: ', results[4], '%')
    if plot is True:
        x = graphic[0]
        y = graphic[1]
        auc = graphic[2]
        sd_auc = graphic[3]
        plt.figure(num=None, figsize=(8,6))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)
        plt.plot(x, y, 'b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (auc, sd_auc), lw=2, alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Hammers FA model mean ROC curve')
        plt.legend(loc="lower right")
        plt.savefig('D:\\Universidade\\5\\Tese\\Escrita\\ROC curves\\Hammers_FA.png', format='png')

    print('\n')

    os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\DTI\\Data\\Hammers\\MD')
    path_1 = fs
    model, results, graphic = avaliar(path_1, classifier, number)
    print('Hammers MD model accuracy: ', results[0], '%')
    print('Hammers MD model sensitivity: ', results[1], '%')
    print('Hammers MD model specificity: ', results[2], '%')
    print('90% of classifiers perform above: ', results[3], '%')
    print('Percentage of full accuracy: ', results[4], '%')
    if plot is True:
        x = graphic[0]
        y = graphic[1]
        auc = graphic[2]
        sd_auc = graphic[3]
        plt.figure(num=None, figsize=(8,6))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)
        plt.plot(x, y, 'b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (auc, sd_auc), lw=2, alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Hammers MD model mean ROC curve')
        plt.legend(loc="lower right")
        plt.savefig('D:\\Universidade\\5\\Tese\\Escrita\\ROC curves\\Hammers_MD.png', format='png')

    print('\n')

    os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\DTI\\Data\\JHU\\FA')
    path_1 = fs
    model, results, graphic = avaliar(path_1, classifier, number)
    print('JHU FA model accuracy: ', results[0], '%')
    print('JHU FA model sensitivity: ', results[1], '%')
    print('JHU FA model specificity: ', results[2], '%')
    print('90% of classifiers perform above: ', results[3], '%')
    print('Percentage of full accuracy: ', results[4], '%')
    if plot is True:
        x = graphic[0]
        y = graphic[1]
        auc = graphic[2]
        sd_auc = graphic[3]
        plt.figure(num=None, figsize=(8,6))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)
        plt.plot(x, y, 'b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (auc, sd_auc), lw=2, alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('JHU FA model mean ROC curve')
        plt.legend(loc="lower right")
        plt.savefig('D:\\Universidade\\5\\Tese\\Escrita\\ROC curves\\JHU_FA.png', format='png')

    print('\n')

    os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\DTI\\Data\\JHU\\MD')
    path_1 = fs
    model, results, graphic = avaliar(path_1, classifier, number)
    print('JHU MD model accuracy: ', results[0], '%')
    print('JHU MD model sensitivity: ', results[1], '%')
    print('JHU MD model specificity: ', results[2], '%')
    print('90% of classifiers perform above: ', results[3], '%')
    print('Percentage of full accuracy: ', results[4], '%')
    if plot is True:
        x = graphic[0]
        y = graphic[1]
        auc = graphic[2]
        sd_auc = graphic[3]
        plt.figure(num=None, figsize=(8,6))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)
        plt.plot(x, y, 'b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (auc, sd_auc), lw=2, alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('JHU MD model mean ROC curve')
        plt.legend(loc="lower right")
        plt.savefig('D:\\Universidade\\5\\Tese\\Escrita\\ROC curves\\JHU_MD.png', format='png')

    print('\n')

    print('PIB')
    os.chdir('D:\\Universidade\\5\Tese\\Classificacao\\PIB\\Data\\Cerebelo')
    path_1 = fs
    model, results, graphic = avaliar(path_1, classifier, number)
    print('SUVR Cerebellum accuracy: ', results[0], '%')
    print('SUVR Cerebellum model sensitivity: ', results[1], '%')
    print('SUVR Cerebellum model specificity: ', results[2], '%')
    print('90% of classifiers perform above: ', results[3], '%')
    print('Percentage of full accuracy: ', results[4], '%')
    if plot is True:
        x = graphic[0]
        y = graphic[1]
        auc = graphic[2]
        sd_auc = graphic[3]
        plt.figure(num=None, figsize=(8,6))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)
        plt.plot(x, y, 'b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (auc, sd_auc), lw=2, alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('SUVR Cerebellum model mean ROC curve')
        plt.legend(loc="lower right")
        plt.savefig('D:\\Universidade\\5\\Tese\\Escrita\\ROC curves\\SUVR_Cerebellum.png', format='png')

    print('\n')

    os.chdir('D:\\Universidade\\5\Tese\\Classificacao\\PIB\\Data\\GM')
    path_1 = fs
    model, results, graphic = avaliar(path_1, classifier, number)
    print('SUVR GM accuracy: ', results[0], '%')
    print('SUVR GM model sensitivity: ', results[1], '%')
    print('SUVR GM model specificity: ', results[2], '%')
    print('90% of classifiers perform above: ', results[3], '%')
    print('Percentage of full accuracy: ', results[4], '%')
    if plot is True:
        x = graphic[0]
        y = graphic[1]
        auc = graphic[2]
        sd_auc = graphic[3]
        plt.figure(num=None, figsize=(8,6))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)
        plt.plot(x, y, 'b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (auc, sd_auc), lw=2, alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('SUVR GM model mean ROC curve')
        plt.legend(loc="lower right")
        plt.savefig('D:\\Universidade\\5\\Tese\\Escrita\\ROC curves\\SUVR_GM.png', format='png')

    print('\n')

    os.chdir('D:\\Universidade\\5\Tese\\Classificacao\\PIB\\Data\\WM')
    path_1 = fs
    model, results, graphic = avaliar(path_1, classifier, number)
    print('SUVR WM accuracy: ', results[0], '%')
    print('SUVR WM model sensitivity: ', results[1], '%')
    print('SUVR WM model specificity: ', results[2], '%')
    print('90% of classifiers perform above: ', results[3], '%')
    print('Percentage of full accuracy: ', results[4], '%')
    if plot is True:
        x = graphic[0]
        y = graphic[1]
        auc = graphic[2]
        sd_auc = graphic[3]
        plt.figure(num=None, figsize=(8,6))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)
        plt.plot(x, y, 'b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (auc, sd_auc), lw=2, alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('SUVR WM model mean ROC curve')
        plt.legend(loc="lower right")
        plt.savefig('D:\\Universidade\\5\\Tese\\Escrita\\ROC curves\\SUVR_WM.png', format='png')

    print('\n')
    plt.show()

    return


def ensemble(p_mri=0.4, p_pib=0.4, p_dti=0.2):
    import statistics as st
    from sklearn.metrics import roc_curve, auc
    import numpy as np
    from scipy import interp

    # Store the values of assembly of the 2000 runs
    #Ensemble
    accuracy_ass_vec = []
    sensitivity_ass_vec = []
    specificity_ass_vec = []
    tprs_ass = []
    aucs_ass = []
    mean_fpr_ass = np.linspace(0, 1, 100)

    #MRI
    accuracy_mri_vec = []
    sensitivity_mri_vec = []
    specificity_mri_vec = []
    tprs_mri = []
    aucs_mri = []
    mean_fpr_mri = np.linspace(0, 1, 100)

    #PIB
    accuracy_pib_vec = []
    sensitivity_pib_vec = []
    specificity_pib_vec = []
    tprs_pib = []
    aucs_pib = []
    mean_fpr_pib = np.linspace(0, 1, 100)

    #DTI
    accuracy_dti_vec = []
    sensitivity_dti_vec = []
    specificity_dti_vec = []
    tprs_dti = []
    aucs_dti = []
    mean_fpr_dti = np.linspace(0, 1, 100)

    # Indices of the complementary data
    os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\')
    indices = np.genfromtxt('Indices.csv', delimiter=';', skip_header=1, dtype=None)
    indice_MRI = indices[:, 0]
    indice_PIB = indices[:, 1]
    indice_DTI = indices[:, 2]

    # MRI data
    os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\MRI\\Estrutural\\Neuro\\GM')
    data_mri = np.genfromtxt('Randf.csv', delimiter=';', skip_header=1)
    data_mri = data_mri[indice_MRI, :]
    data_mri, label = normal_data(data_mri)

    # PIB data
    os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\PIB\\Data\\WM')
    data_pib = np.genfromtxt('Randf.csv', delimiter=';', skip_header=1)
    data_pib = data_pib[indice_PIB, :]
    data_pib, label = normal_data(data_pib)

    # DTI data
    os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\DTI\\Data\\Desikan\\FA')
    data_dti = np.genfromtxt('Randf.csv', delimiter=';', skip_header=1)
    data_dti = data_dti[indice_DTI, :]
    data_dti, label = normal_data(data_dti)

    for j in range(100):
        print(j)
        # Creating the individual classifier's
        if p_mri > 0:
            # Best model for MRI
            os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\MRI\\Estrutural\\Neuro\\GM')
            model = avaliar('Randf.csv', number=100)
            model_mri = model[0]

            # MRI prediction
            predict_mri = model_mri.predict(data_mri)
            tn, fp, fn, tp = confusion_matrix(label, predict_mri).ravel()
            accuracy = (tn + tp) / (tn + fp + fn + tp)
            accuracy_mri_vec.append(accuracy)
            sensitivity = tp / (tp + fn)
            sensitivity_mri_vec.append(sensitivity)
            specificity = tn / (tn + fp)
            specificity_mri_vec.append(specificity)

            predict_mri = model_mri.predict_proba(data_mri)
            predict_mri = np.array(predict_mri[:, 1])
            fpr, tpr, thresholds = roc_curve(label, predict_mri)
            roc_auc = auc(fpr, tpr)
            tprs_mri.append(interp(mean_fpr_mri, fpr, tpr))
            tprs_mri[-1][0] = 0.0
            aucs_mri.append(roc_auc)

        else:
            predict_mri = np.zeros(len(label))

        if p_pib > 0:
            # Best model for PIB
            os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\PIB\\Data\\WM')
            model = avaliar('Randf.csv', number=100)
            model_pib = model[0]

            # PIB prediction
            predict_pib = model_pib.predict(data_pib)
            tn, fp, fn, tp = confusion_matrix(label, predict_pib).ravel()
            accuracy = (tn + tp) / (tn + fp + fn + tp)
            accuracy_pib_vec.append(accuracy)
            sensitivity = tp / (tp + fn)
            sensitivity_pib_vec.append(sensitivity)
            specificity = tn / (tn + fp)
            specificity_pib_vec.append(specificity)

            predict_pib = model_pib.predict_proba(data_pib)
            predict_pib = np.array(predict_pib[:, 1])
            fpr, tpr, thresholds = roc_curve(label, predict_pib)
            roc_auc = auc(fpr, tpr)
            tprs_pib.append(interp(mean_fpr_pib, fpr, tpr))
            tprs_pib[-1][0] = 0.0
            aucs_pib.append(roc_auc)
        else:
            predict_pib = np.zeros(len(label))

        if p_dti > 0:
            # Best model for DTI
            os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\DTI\\Data\\Desikan\\FA')
            model = avaliar('Randf.csv', number=100)
            model_dti = model[0]

            # DTI prediction
            predict_dti = model_dti.predict(data_dti)
            tn, fp, fn, tp = confusion_matrix(label, predict_dti).ravel()
            accuracy = (tn + tp) / (tn + fp + fn + tp)
            accuracy_dti_vec.append(accuracy)
            sensitivity = tp / (tp + fn)
            sensitivity_dti_vec.append(sensitivity)
            specificity = tn / (tn + fp)
            specificity_dti_vec.append(specificity)

            predict_dti = model_dti.predict_proba(data_dti)
            predict_dti = np.array(predict_dti[:, 1])
            fpr, tpr, thresholds = roc_curve(label, predict_dti)
            roc_auc = auc(fpr, tpr)
            tprs_dti.append(interp(mean_fpr_dti, fpr, tpr))
            tprs_dti[-1][0] = 0.0
            aucs_dti.append(roc_auc)

        else:
            predict_dti = np.zeros(len(label))

        # Ensemble the classifiers predictions
        predict = []
        predict_prob = []
        for i in range(len(predict_mri)):
            value = (predict_mri[i] * p_mri) + (predict_pib[i] * p_pib) + (predict_dti[i] * p_dti)
            predict_prob.append(value)
            value = int(round(value))
            predict.append(value)

        # Results
        tn, fp, fn, tp = confusion_matrix(label, predict).ravel()
        accuracy = (tn + tp) / (tn + fp + fn + tp)
        accuracy_ass_vec.append(accuracy)
        sensitivity = tp / (tp + fn)
        sensitivity_ass_vec.append(sensitivity)
        specificity = tn / (tn + fp)
        specificity_ass_vec.append(specificity)

        fpr, tpr, thresholds = roc_curve(label, predict_prob)
        roc_auc = auc(fpr, tpr)
        tprs_ass.append(interp(mean_fpr_ass, fpr, tpr))
        tprs_ass[-1][0] = 0.0
        aucs_ass.append(roc_auc)

    if p_mri > 0:
        accuracy = st.mean(accuracy_mri_vec) * 100
        print('Mean accuracy from MRI: ', accuracy, ' %')
        sensitivity = st.mean(sensitivity_mri_vec) * 100
        print('Mean sensitivity from MRI: ', sensitivity, ' %')
        specificity = st.mean(specificity_mri_vec) * 100
        print('Mean specificity from MRI: ', specificity, ' %')
        print('\n')

        #Graphic
        mean_tpr_mri = np.mean(tprs_mri, axis=0)
        mean_tpr_mri[-1] = 1.0
        mean_auc_mri = auc(mean_fpr_mri, mean_tpr_mri)
        std_auc_mri = np.std(aucs_mri)
        plt.plot(mean_fpr_mri, mean_tpr_mri, 'b', label=r'Mean ROC MRI-based classifier (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_mri, std_auc_mri), lw=2, alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])


    if p_pib > 0:
        accuracy = st.mean(accuracy_pib_vec) * 100
        print('Mean accuracy from PIB: ', accuracy, ' %')
        sensitivity = st.mean(sensitivity_pib_vec) * 100
        print('Mean sensitivity from PIB: ', sensitivity, ' %')
        specificity = st.mean(specificity_pib_vec) * 100
        print('Mean specificity from PIB: ', specificity, ' %')
        print('\n')

        #Graphic
        mean_tpr_pib = np.mean(tprs_pib, axis=0)
        mean_tpr_pib[-1] = 1.0
        mean_auc_pib = auc(mean_fpr_pib, mean_tpr_pib)
        std_auc_pib = np.std(aucs_pib)
        plt.plot(mean_fpr_pib, mean_tpr_pib, 'r', label=r'Mean ROC PIB-based classifier (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_pib, std_auc_pib), lw=2, alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])

    if p_dti > 0:
        accuracy = st.mean(accuracy_dti_vec) * 100
        print('Mean accuracy from DTI: ', accuracy, ' %')
        sensitivity = st.mean(sensitivity_dti_vec) * 100
        print('Mean sensitivity from DTI: ', sensitivity, ' %')
        specificity = st.mean(specificity_dti_vec) * 100
        print('Mean specificity from DTI: ', specificity, ' %')
        print('\n')

        #Graphic
        mean_tpr_dti = np.mean(tprs_dti, axis=0)
        mean_tpr_dti[-1] = 1.0
        mean_auc_dti = auc(mean_fpr_dti, mean_tpr_dti)
        std_auc_dti = np.std(aucs_dti)
        plt.plot(mean_fpr_dti, mean_tpr_dti, 'y', label=r'Mean ROC DTI-based classifier (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_dti, std_auc_dti), lw=2, alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])

    accuracy = st.mean(accuracy_ass_vec) * 100
    print('Mean accuracy from assembly: ', accuracy, ' %')
    sensitivity = st.mean(sensitivity_ass_vec) * 100
    print('Mean sensitivity from assembly: ', sensitivity, ' %')
    specificity = st.mean(specificity_ass_vec) * 100
    print('Mean specificity from assembly: ', specificity, ' %')
    print('\n')

    #Graphic
    mean_tpr_ass = np.mean(tprs_ass, axis=0)
    mean_tpr_ass[-1] = 1.0
    mean_auc_ass = auc(mean_fpr_ass, mean_tpr_ass)
    std_auc_ass = np.std(aucs_ass)
    plt.plot(mean_fpr_ass, mean_tpr_ass,
             label=r'Mean ROC Ensemble (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_ass, std_auc_ass), lw=2, alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)
    plt.title('ROC curve comparison')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()
    return


def pib_ext():
    import statistics as st
    from sklearn.metrics import roc_curve, auc
    import numpy as np
    from scipy import interp

    # External data
    os.chdir('D:\\Universidade\\5\\Tese\\External Validation\\Final_Data')
    data_pib = np.genfromtxt('EV_PIB3.csv', delimiter=';', skip_header=1)
    data_pib, label = normal_data(data_pib)

    accuracy_vec = []
    baccuracy_vec = []
    sensitivity_vec = []
    specificity_vec = []
    tprs_vec = []
    aucs_vec = []
    mean_fpr_vec = np.linspace(0, 1, 100)

    for i in range(1000):
        # Best model for PIB
        os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\PIB\\Data\\WM')
        model = avaliar('Randf.csv', number=100)
        model_pib = model[0]

        # PIB prediction
        predict_pib = model_pib.predict(data_pib)
        tn, fp, fn, tp = confusion_matrix(label, predict_pib).ravel()
        accuracy = (tn + tp) / (tn + fp + fn + tp)
        accuracy_vec.append(accuracy)
        sensitivity = tp / (tp + fn)
        sensitivity_vec.append(sensitivity)
        specificity = tn / (tn + fp)
        specificity_vec.append(specificity)
        p = np.sum(label)
        n = len(label) - p
        baccuracy = ((tp / p) + (tn / n)) / 2
        baccuracy_vec.append(baccuracy)

        predict_pib_prob = model_pib.predict_proba(data_pib)
        predict_pib_prob = predict_pib_prob[:, 1]
        fpr, tpr, thresholds = roc_curve(label, predict_pib_prob)
        roc_auc = auc(fpr, tpr)
        tprs_vec.append(interp(mean_fpr_vec, fpr, tpr))
        tprs_vec[-1][0] = 0.0
        aucs_vec.append(roc_auc)


    accuracy = st.mean(accuracy_vec) * 100
    print('Mean accuracy from external PIB: ', accuracy, ' %')
    sensitivity = st.mean(sensitivity_vec) * 100
    print('Mean sensitivity from external PIB: ', sensitivity, ' %')
    specificity = st.mean(specificity_vec) * 100
    print('Mean specificity from external PIB: ', specificity, ' %')
    baccuracy = st.mean(baccuracy_vec) * 100
    print('Mean balanced accuracy from external PIB: ', baccuracy, ' %')
    print('\n')

    #Graphic
    mean_tpr_vec = np.mean(tprs_vec, axis=0)
    mean_tpr_vec[-1] = 1.0
    mean_auc_vec = auc(mean_fpr_vec, mean_tpr_vec)
    std_auc_vec = np.std(aucs_vec)
    plt.plot(mean_fpr_vec, mean_tpr_vec,
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_vec, std_auc_vec), lw=2, alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)
    plt.title('SUVR WM EBM')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

    return


def mri_ext():
    import statistics as st
    from sklearn.metrics import roc_curve, auc
    import numpy as np
    from scipy import interp

    # External data
    os.chdir('D:\\Universidade\\5\\Tese\\External Validation\\Final_Data')
    data_mri = np.genfromtxt('EV_MRI.csv', delimiter=';', skip_header=1)
    data, label = normal_data(data_mri)

    accuracy_vec = []
    baccuracy_vec = []
    sensitivity_vec = []
    specificity_vec = []
    tprs_vec = []
    aucs_vec = []
    mean_fpr_vec = np.linspace(0, 1, 100)

    for i in range(1000):
        # Best model for MRI
        os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\MRI\\Estrutural\\Neuro\\GM')
        model = avaliar('Randf.csv', number=100)
        model_mri = model[0]

        # MRI prediction
        predict_mri = model_mri.predict(data)
        tn, fp, fn, tp = confusion_matrix(label, predict_mri).ravel()
        accuracy = (tn + tp) / (tn + fp + fn + tp)
        accuracy_vec.append(accuracy)
        sensitivity = tp / (tp + fn)
        sensitivity_vec.append(sensitivity)
        specificity = tn / (tn + fp)
        specificity_vec.append(specificity)
        p = np.sum(label)
        n = len(label) - p
        baccuracy = ((tp / p) + (tn / n)) / 2
        baccuracy_vec.append(baccuracy)

        predict_mri_prob = model_mri.predict_proba(data)
        predict_mri_prob = predict_mri_prob[:, 1]
        fpr, tpr, thresholds = roc_curve(label, predict_mri_prob)
        roc_auc = auc(fpr, tpr)
        tprs_vec.append(interp(mean_fpr_vec, fpr, tpr))
        tprs_vec[-1][0] = 0.0
        aucs_vec.append(roc_auc)

    accuracy = st.mean(accuracy_vec) * 100
    print('Mean accuracy from external MRI: ', accuracy, ' %')
    sensitivity = st.mean(sensitivity_vec) * 100
    print('Mean sensitivity from external MRI: ', sensitivity, ' %')
    specificity = st.mean(specificity_vec) * 100
    print('Mean specificity from external MRI: ', specificity, ' %')
    baccuracy = st.mean(baccuracy_vec) * 100
    print('Mean balanced accuracy from external MRI: ', baccuracy, ' %')
    print('\n')

    #Graphic
    mean_tpr_vec = np.mean(tprs_vec, axis=0)
    mean_tpr_vec[-1] = 1.0
    mean_auc_vec = auc(mean_fpr_vec, mean_tpr_vec)
    std_auc_vec = np.std(aucs_vec)
    plt.plot(mean_fpr_vec, mean_tpr_vec,
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_vec, std_auc_vec), lw=2, alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)
    plt.title('Neuromorphometrics GM EBM')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()

    return


def dti_ext(p_mri=0.5, p_dti=0.5):
    import statistics as st
    from sklearn.metrics import roc_curve, auc
    import numpy as np
    from scipy import interp

    accuracy_vec_dti = []
    sensitivity_vec_dti = []
    specificity_vec_dti = []
    baccuracy_vec_dti = []
    tprs_dti = []
    aucs_dti = []
    mean_fpr_dti = np.linspace(0, 1, 100)

    accuracy_vec_mri = []
    baccuracy_vec_mri = []
    sensitivity_vec_mri = []
    specificity_vec_mri = []
    tprs_mri = []
    aucs_mri = []
    mean_fpr_mri = np.linspace(0, 1, 100)


    accuracy_vec = []
    sensitivity_vec = []
    specificity_vec = []
    baccuracy_vec = []
    tprs_vec = []
    aucs_vec = []
    mean_fpr_vec = np.linspace(0, 1, 100)

    # External data
    os.chdir('D:\\Universidade\\5\\Tese\\External Validation\\Final_Data')
    data_dti = np.genfromtxt('EV_DTI.csv', delimiter=';', skip_header=1)
    data_dti, label = normal_data(data_dti)

    os.chdir('D:\\Universidade\\5\\Tese\\External Validation\\Final_Data')
    data_mri = np.genfromtxt('EV_MRI_DTI.csv', delimiter=';', skip_header=1)
    data_mri, label = normal_data(data_mri)

    for i in range(1000):
        if p_dti > 0:
            # Best model for DTI
            os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\DTI\\Data\\Desikan\\FA')
            model = avaliar('Randf.csv', number=100)
            model_dti = model[0]

            # DTI prediction
            predict_dti = model_dti.predict(data_dti)
            tn, fp, fn, tp = confusion_matrix(label, predict_dti).ravel()
            accuracy = (tn + tp) / (tn + fp + fn + tp)
            accuracy_vec_dti.append(accuracy)
            sensitivity = tp / (tp + fn)
            sensitivity_vec_dti.append(sensitivity)
            specificity = tn / (tn + fp)
            specificity_vec_dti.append(specificity)
            p = np.sum(label)
            n = len(label) - p
            baccuracy = ((tp / p) + (tn / n)) / 2
            baccuracy_vec_dti.append(baccuracy)

            predict_dti = model_dti.predict_proba(data_dti)
            predict_dti = np.array(predict_dti[:, 1])

            fpr, tpr, thresholds = roc_curve(label, predict_dti)
            roc_auc = auc(fpr, tpr)
            tprs_dti.append(interp(mean_fpr_dti, fpr, tpr))
            tprs_dti[-1][0] = 0.0
            aucs_dti.append(roc_auc)
        else:
            predict_dti = np.zeros(len(label))

        if p_mri > 0:
            # Best model for MRI
            os.chdir('D:\\Universidade\\5\\Tese\\Classificacao\\MRI\\Estrutural\\Neuro\\GM')
            model = avaliar('Randf.csv', number=100)
            model_mri = model[0]

            # MRI prediction
            predict_mri = model_mri.predict(data_mri)
            tn, fp, fn, tp = confusion_matrix(label, predict_mri).ravel()
            accuracy = (tn + tp) / (tn + fp + fn + tp)
            accuracy_vec_mri.append(accuracy)
            sensitivity = tp / (tp + fn)
            sensitivity_vec_mri.append(sensitivity)
            specificity = tn / (tn + fp)
            specificity_vec_mri.append(specificity)
            p = np.sum(label)
            n = len(label) - p
            baccuracy = ((tp / p) + (tn / n)) / 2
            baccuracy_vec_mri.append(baccuracy)

            predict_mri = model_mri.predict_proba(data_mri)
            predict_mri = np.array(predict_mri[:, 1])
            fpr, tpr, thresholds = roc_curve(label, predict_mri)
            roc_auc = auc(fpr, tpr)
            tprs_mri.append(interp(mean_fpr_mri, fpr, tpr))
            tprs_mri[-1][0] = 0.0
            aucs_mri.append(roc_auc)

        else:
            predict_mri = np.zeros(len(label))

        # Ensemble
        predict = []
        predict_prob = []
        for j in range(len(predict_mri)):
            value = (predict_mri[j] * p_mri) + (predict_dti[j] * p_dti)
            predict_prob.append(value)
            value = int(round(value))
            predict.append(value)

        # Results
        tn, fp, fn, tp = confusion_matrix(label, predict).ravel()
        accuracy = (tn + tp) / (tn + fp + fn + tp)
        accuracy_vec.append(accuracy)
        sensitivity = tp / (tp + fn)
        sensitivity_vec.append(sensitivity)
        specificity = tn / (tn + fp)
        specificity_vec.append(specificity)
        p = np.sum(label)
        n = len(label) - p
        baccuracy = ((tp / p) + (tn / n)) / 2
        baccuracy_vec.append(baccuracy)

        fpr, tpr, thresholds = roc_curve(label, predict_prob)
        roc_auc = auc(fpr, tpr)
        tprs_vec.append(interp(mean_fpr_vec, fpr, tpr))
        tprs_vec[-1][0] = 0.0
        aucs_vec.append(roc_auc)

    if p_dti > 0:
        accuracy = st.mean(accuracy_vec_dti) * 100
        print('Mean accuracy from external DTI: ', accuracy, ' %')
        sensitivity = st.mean(sensitivity_vec_dti) * 100
        print('Mean sensitivity from external DTI: ', sensitivity, ' %')
        specificity = st.mean(specificity_vec_dti) * 100
        print('Mean specificity from external DTI: ', specificity, ' %')
        baccuracy = st.mean(baccuracy_vec_dti) * 100
        print('Mean balanced accuracy from external DTI: ', baccuracy, ' %')
        print('\n')

        #Graphic
        mean_tpr_dti = np.mean(tprs_dti, axis=0)
        mean_tpr_dti[-1] = 1.0
        mean_auc_dti = auc(mean_fpr_dti, mean_tpr_dti)
        std_auc_dti = np.std(aucs_dti)
        if p_mri > 0:
            plt.plot(mean_fpr_dti, mean_tpr_dti, 'y', label=r'Mean ROC DTI-based classifier (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_dti, std_auc_dti), lw=2, alpha=.8)
        else:
            plt.plot(mean_fpr_dti, mean_tpr_dti, 'y',
                     label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_dti, std_auc_dti), lw=2, alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)
        plt.title('Destrieux MD FBM')
        plt.legend(loc="lower right")

    if p_mri > 0:
        accuracy = st.mean(accuracy_vec_mri) * 100
        print('Mean accuracy from external MRI: ', accuracy, ' %')
        sensitivity = st.mean(sensitivity_vec_mri) * 100
        print('Mean sensitivity from external MRI: ', sensitivity, ' %')
        specificity = st.mean(specificity_vec_mri) * 100
        print('Mean specificity from external MRI: ', specificity, ' %')
        baccuracy = st.mean(baccuracy_vec_mri) * 100
        print('Mean balanced accuracy from external MRI: ', baccuracy, ' %')
        print('\n')

        #Graphic
        mean_tpr_mri = np.mean(tprs_mri, axis=0)
        mean_tpr_mri[-1] = 1.0
        mean_auc_mri = auc(mean_fpr_mri, mean_tpr_mri)
        std_auc_mri = np.std(aucs_mri)
        plt.plot(mean_fpr_mri, mean_tpr_mri, 'b', label=r'Mean ROC MRI (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_mri, std_auc_mri), lw=2, alpha=.8)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])

    accuracy = st.mean(accuracy_vec) * 100
    print('Mean accuracy from external Ensemble: ', accuracy, ' %')
    sensitivity = st.mean(sensitivity_vec) * 100
    print('Mean sensitivity from external Ensemble: ', sensitivity, ' %')
    specificity = st.mean(specificity_vec) * 100
    print('Mean specificity from external Ensemble: ', specificity, ' %')
    baccuracy = st.mean(baccuracy_vec) * 100
    print('Mean balanced accuracy from external Ensemble: ', baccuracy, ' %')
    print('\n')

    #Graphic
    mean_tpr_vec = np.mean(tprs_vec, axis=0)
    mean_tpr_vec[-1] = 1.0
    mean_auc_vec = auc(mean_fpr_vec, mean_tpr_vec)
    std_auc_vec = np.std(aucs_vec)
    plt.plot(mean_fpr_vec, mean_tpr_vec,
             label=r'Mean ROC Ensemble (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_vec, std_auc_vec), lw=2, alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)
    plt.title('ROC curve comparison')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()


    return



#evaluation('lgr', 2000, 'Filter.csv', plot=True)
ensemble(p_mri=0, p_pib=0.5, p_dti=0.5)
#ib_ext()
#mri_ext()
#dti_ext()


