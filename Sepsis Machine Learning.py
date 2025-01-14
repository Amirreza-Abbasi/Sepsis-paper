import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, roc_curve, precision_recall_curve
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
import matplotlib.pyplot as plt




# =============================================================================
# Machine learning
# =============================================================================
def ML(df, Dataset_Code, DataType):
    
    df = df[df['Group'].str.lower().isin(['c', 'p', 'C', 'P'])]

    # df = Z_df.copy()
    # DataType = 'Zlog'
    # Method_Title = DataType
    
    
    X_data = df.drop('Group', axis=1)
    Y_data = df['Group']
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, shuffle=True)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train).astype('float32')
    X_test = scaler.fit_transform(X_test).astype('float32')
    
    
    # Plot Confusion Matrix
    def CM_Plot(clf, Method_Title, Result_df):
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        # Confusion Matrix
        confusion_matrix = confusion_matrix(Result_df['True'], Result_df['Predicted'])
        ConfusionMatrixDisplay = ConfusionMatrixDisplay(confusion_matrix)#.plot()
        TP = confusion_matrix[0, 0]
        FN = confusion_matrix[0, 1]
        FP = confusion_matrix[1, 0]
        TN = confusion_matrix[1, 1]
        
        plt.rcParams.update(plt.rcParamsDefault)
        fig = plt.gcf()
        fig.set_size_inches(5, 5)
        plt.clf()
        ConfusionMatrixDisplay.plot()
        plt.title(f'{DataType} - {Method_Title} Confusion Matrix', fontsize=12, color = 'red')
        plt.savefig(f'Machine Learning\{Dataset_Code}\{DataType} - {Method_Title}_{Dataset_Code}_CM_Results.png', dpi=200 )
        plt.show()
        return(TP, FN, FP, TN)
    
    # Plot Curve Results
    def Curves_Plot(clf, Method_Title):
        
        if Method_Title == 'DTC':
            y_score = Result_df['Predicted']
        else:
            y_score = clf.decision_function(X_test) 
        
        # Roc Curve
        fpr, tpr, _ = roc_curve(Y_test, y_score, pos_label=clf.classes_[1])
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        # Precision Recall
        prec, recall, _ = precision_recall_curve(Y_test, y_score, pos_label=clf.classes_[1])
        pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()
        
        plt.rcParams.update(plt.rcParamsDefault)
        plt.clf()
        fig, axs = plt.subplots(1, 2, figsize=(10, 6))
        roc_display.plot(axs[0])
        axs[0].title.set_text('Roc Curve')
        pr_display.plot(axs[1])
        axs[1].title.set_text('Precision Recall')
        plt.suptitle(f'{DataType} - {Method_Title}', fontsize=15, color = 'red')
        plt.savefig(f'Machine Learning\{Dataset_Code}\{DataType} - {Method_Title}_{Dataset_Code}_Curve.png', dpi=200 )
    
    
    
    # Logistic Regression
    def LogisticRegression(Method_Title):
        print('\nTraining LogisticRegression\n')
        global Result_df, lr_acc
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(random_state=0)
        clf.fit(X_train, Y_train)
        clf_pred = clf.predict(X_test)
        
        lr_acc = round(accuracy_score(Y_test, clf.predict(X_test)), 2)
        print('[0]Logistic Regression Training Score:', clf.score(X_train, Y_train))
        print('[0]Logistic Regression Test Accuracy:', accuracy_score(Y_test, clf.predict(X_test)))
        cr = classification_report(Y_test, clf.predict(X_test))
        print('[0]Logistic Regression Classification Report:\n', cr)
        
        Result_df = pd.DataFrame(clf_pred)
        Result_df = pd.concat([Result_df, Y_test.reset_index(drop=True)], axis=1).reset_index(drop=True)
        Result_df.columns = ['Predicted', 'True']
        Result_df['Accuracy'] = np.where(Result_df['Predicted'] == Result_df['True'], True, False).astype(int)
        Result_df.to_csv(f'Machine Learning\{Dataset_Code}\{DataType}_{Dataset_Code}_RL_prediction_Result.csv' , index=False)
        
        print('Accuracy: ', Result_df['Accuracy'].sum() / len(Result_df) * 100, '%')
            
        temp = f'{Method_Title}\n\n{cr}'
        text_file = open(f'Machine Learning\{Dataset_Code}\{DataType}_{Dataset_Code}_LR_Results.txt', "w")
        text_file.write(temp)
        text_file.close()
        
        TP, FN, FP, TN = CM_Plot(clf, Method_Title, Result_df)
        Curves_Plot(clf, Method_Title)
        return(TP, FN, FP, TN)
    
    # Support Vector Machine
    def SupportVectorMachine(Method_Title):
        print('\nTraining SupportVectorMachine\n')
        global Result_df, svm_acc
        from sklearn import svm
        clf = svm.SVC()
        clf.fit(X_train, Y_train)
        clf_pred = clf.predict(X_test)
        
        svm_acc = round(accuracy_score(Y_test, clf.predict(X_test)), 2)
        print('[0]Support vector machine Training Score:', clf.score(X_train, Y_train))
        print('[0]Support vector machine Test Accuracy:', accuracy_score(Y_test, clf.predict(X_test)))
        cr = classification_report(Y_test, clf.predict(X_test))
        print('[0]Support vector machine Classification Report:\n', cr)
        
        Result_df = pd.DataFrame(clf_pred)
        Result_df = pd.concat([Result_df, Y_test.reset_index(drop=True)], axis=1).reset_index(drop=True)
        Result_df.columns = ['Predicted', 'True']
        
        Result_df['Accuracy'] = np.where(Result_df['Predicted'] == Result_df['True'], True, False).astype(int)
        Result_df.to_csv(f'Machine Learning\{Dataset_Code}\{DataType}_{Dataset_Code}_SVM_prediction_Result.csv' , index=False)
        
        print('Accuracy: ', Result_df['Accuracy'].sum() / len(Result_df) * 100, '%')
        
        temp = f'{Method_Title}\n\n{cr}'
        text_file = open(f'Machine Learning\{Dataset_Code}\{DataType}_{Dataset_Code}_SVM_Results.txt', "w")
        text_file.write(temp)
        text_file.close()
        
        TP, FN, FP, TN = CM_Plot(clf, Method_Title, Result_df)
        Curves_Plot(clf, Method_Title)
        return(TP, FN, FP, TN)
    
    # Gradient Boosting Classifier
    def GradientBoostingClassifier(Method_Title):
        print('\nTraining GradientBoostingClassifier\n')
        global Result_df, gbm_acc
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, Y_train)
        clf.score(X_test, Y_test)
        clf_pred = clf.predict(X_test)
        
        gbm_acc = round(accuracy_score(Y_test, clf.predict(X_test)), 2)
        print('[0]Gradient Boosting Classifier Training Score:', clf.score(X_train, Y_train))
        print('[0]Gradient Boosting Classifier Test Accuracy:', accuracy_score(Y_test, clf.predict(X_test)))
        cr = classification_report(Y_test, clf.predict(X_test))
        print('[0]Gradient Boosting Classifier Report:\n', cr)
        
        Result_df = pd.DataFrame(clf_pred)
        Result_df = pd.concat([Result_df, Y_test.reset_index(drop=True)], axis=1).reset_index(drop=True)
        Result_df.columns = ['Predicted', 'True']
        
        Result_df['Accuracy'] = np.where(Result_df['Predicted'] == Result_df['True'], True, False).astype(int)
        Result_df.to_csv(f'Machine Learning\{Dataset_Code}\{DataType}_{Dataset_Code}_GBM_prediction_Result.csv' , index=False)
        
        print('Accuracy: ', Result_df['Accuracy'].sum() / len(Result_df) * 100, '%')
    
        temp = f'{Method_Title}\n\n{cr}'
        text_file = open(f'Machine Learning\{Dataset_Code}\{DataType}_{Dataset_Code}_GBM_Results.txt', "w")
        text_file.write(temp)
        text_file.close()
        
        TP, FN, FP, TN = CM_Plot(clf, Method_Title, Result_df)
        Curves_Plot(clf, Method_Title)
        return(TP, FN, FP, TN)
    
    # Decision Tree Classifier
    def DecisionTreeClassifier(Method_Title):
        print('\nTraining DecisionTreeClassifier\n')
        global Result_df, dt_acc
        # from sklearn.model_selection import cross_val_score
        from sklearn.tree import DecisionTreeClassifier
        
        clf = DecisionTreeClassifier(random_state=0).fit(X_train, Y_train)
        # clf_Score = cross_val_score(clf, X_test, Y_test, cv=10)
        clf_pred = clf.predict(X_test)
        
        dt_acc = round(accuracy_score(Y_test, clf.predict(X_test)), 2)
        print('[0]Decision Tree Classifier Training Score:', clf.score(X_train, Y_train))
        print('[0]Decision Tree Classifier Test Accuracy:', accuracy_score(Y_test, clf.predict(X_test)))
        cr = classification_report(Y_test, clf.predict(X_test))
        print('[0]Decision Tree Classifier Report:\n', cr)
        
        Result_df = pd.DataFrame(clf_pred)
        Result_df = pd.concat([Result_df, Y_test.reset_index(drop=True)], axis=1).reset_index(drop=True)
        Result_df.columns = ['Predicted', 'True']
        
        Result_df['Accuracy'] = np.where(Result_df['Predicted'] == Result_df['True'], True, False).astype(int)
        Result_df.to_csv(f'Machine Learning\{Dataset_Code}\{DataType}_{Dataset_Code}_DTC_prediction_Result.csv' , index=False)
        
        print('Accuracy: ', Result_df['Accuracy'].sum() / len(Result_df) * 100, '%')
    
        temp = f'{Method_Title}\n\n{cr}'
        text_file = open(f'Machine Learning\{Dataset_Code}\{DataType}_{Dataset_Code}_DTC_Results.txt', "w")
        text_file.write(temp)
        text_file.close()
        
        TP, FN, FP, TN = CM_Plot(clf, Method_Title, Result_df)
        # Curves_Plot(clf, Method_Title)
        return(TP, FN, FP, TN)
    
    LogisticRegression('LR')
    SupportVectorMachine('SVM')
    GradientBoostingClassifier('GBC')
    DecisionTreeClassifier('DTC')


Dataset_Code = 'GSE28750'

# ============ Edit the groups to c (Control) or p (Patient) and leave the rest

Z_df = pd.read_csv(f'./Datasets/{Dataset_Code}_Zlog_Ready_to_Train.csv', header=0, index_col=0)
Target_df = pd.read_csv(f'./Datasets/{Dataset_Code}_Target_Genes_Ready_to_Train.csv', header=0, index_col=0)


# ==== Change control, patient to 0,1 befor training =========
ML(Z_df, Dataset_Code, 'Zlog')
ML(Target_df, Dataset_Code, 'Target Genes')

df = Z_df.copy()
DataType = 'Zlog'
