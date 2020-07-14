from flask import Flask, render_template, send_file
import numpy as np
import matplotlib as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FC

app = Flask(__name__)

@app.route("/mypic")
def mypic():
    return render_template("mypic.html")

@app.route("/plot")
def plot():
    fig,axis = plt.subplots(1)
    x=list(range(1,6))
    y=list([0,2,1,3,4])
    axis.plot(x,y)

    canvas = FC(fig)

    from io import BytesIO

    img = BytesIO()
    fig.savefig(img)
    img.seek(0)

    plt.show()
    return send_file(img, mimetype="image/png")


if __name__ == "__main__":
    port = 5050
    app.debug = False
    app.run(port = port)
    
