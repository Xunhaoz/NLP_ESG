import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score

if __name__ == '__main__':
    df = pd.read_csv("Bert Model/check.csv")
    precision = precision_score(df['impact_type'], df['predict'], average='macro')
    print(f'Macro precision: {precision}')

    precision = precision_score(df['impact_type'], df['predict'], average='micro')
    print(f'Precision: {precision}')

    cm = confusion_matrix(df['impact_type'], df['predict'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3, 4])
    disp.plot()
    plt.show()
