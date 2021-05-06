import matplotlib.pyplot as plt

def graph(data, x_label, y_label) :
    x_data = data[x_label].values
    y_data = data[y_label].values
    plt.plot(x_data, y_data, color = 'b')
    plt.show()


