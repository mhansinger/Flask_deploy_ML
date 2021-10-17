'''
Flask app for model comparison
'''
from flask import Flask
from flask import Response
from flask_restful import reqparse
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import joblib
import io
import numpy as np
from sklearn.metrics import mean_squared_error

# create api
app = Flask(__name__)

# Create test dataset
rng1 = np.random.RandomState(0)
rng2 = np.random.RandomState(1)
X_train = np.linspace(0, 6, 100)[:, np.newaxis]
y_train = np.sin(X_train).ravel() + np.sin(6 * X_train).ravel() + rng1.normal(0, 0.1, X_train.shape[0])
X_test = np.linspace(0, 6, 100)[:, np.newaxis]
y_test = np.sin(X_test).ravel() + np.sin(6 * X_test).ravel() + rng2.normal(0, 0.1, X_test.shape[0])


def load_models():
    # load pretrained models: sklearn lib is not needed!
    model1 = joblib.load('models/regr_1.mdl')
    model2 = joblib.load('models/regr_2.mdl')

    return model1, model2

@app.route('/staticpredict', methods=['GET'])
def predict_data():
    '''get MSE values for the two predictions

    :return: json
    '''

    model1, model2 = load_models()

    y_1 = model1.predict(X_test)
    y_2 = model2.predict(X_test)

    out = {'X': X_test.flatten().tolist(), 'y': y_test.tolist(), 'y_1': y_1.tolist(), 'y_2': y_2.tolist()}

    return out

@app.route('/mse', methods=['GET'])
def predict_mse():
    '''Prediction on the test data set

    :return: json
    '''
    model1, model2 = load_models()

    y_1 = model1.predict(X_test)
    y_2 = model2.predict(X_test)

    mse_out = {'mse_1': mean_squared_error(y_test,y_1), 'mse_2': mean_squared_error(y_test,y_2)}

    return mse_out

@app.route('/print-plot')
def plot_png():
    '''visualize the predictions in the browser'''

    # get the prediction
    out = predict_data()
    y_1 = np.array(out['y_1'])
    y_2 = np.array(out['y_2'])

    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)

    axis.scatter(X_test, y_test, c="k", label="test samples")
    axis.plot(X_test, y_1, c="g", label="model_1", linewidth=2)
    axis.plot(X_test, y_2, c="r", label="model_2", linewidth=2)
    axis.set_xlabel("data")
    axis.set_ylabel("target")
    axis.set_title("Boosted Decision Tree Regression")
    axis.legend()

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/retrain_model', methods=['POST'])
def retrain_model():
    '''retrain the model with new hyper parameters

    :return: json
    '''

    # get parsed values
    parser = reqparse.RequestParser()
    parser.add_argument('max_depth')
    parser.add_argument('n_estimators')
    args = parser.parse_args()  # creates dict
    print(int(args['n_estimators']))

    # Fit regression model
    regr_1 = DecisionTreeRegressor(max_depth=int(args['max_depth']))

    regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=int(args['max_depth'])),
                               n_estimators=int(args['n_estimators']), random_state=rng2)

    #re-train the model
    #try:
    regr_1.fit(X_train, y_train)
    regr_2.fit(X_train, y_train)

    # serialize model
    joblib.dump(regr_1, 'models/regr_1.mdl')
    joblib.dump(regr_2, 'models/regr_2.mdl')

    return {'Success retrain': True}

    # except:
    #    return {'Success retrain': False}


if __name__ == '__main__':
    app.run(debug=True)