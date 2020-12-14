# ML
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def polinomial2_regression (X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state=2)

    # Calling
    lr = fit_poly(X_train,y_train,2)

    # Variables regression 
    print(f'Coeficients:{lr.coef_}, \nIntercept:{lr.intercept_}')

    # The function
    b0,b1,b2 = lr.coef_
    b0 = lr.intercept_
    lr_func = lambda x: b0*x**0 + b1*x**1 + b2*x**2
    print(f"y = {b0}x^0+{b1}x^1+{b2}x^2")

    # plotting on train 
    fig, ax = plt.subplots(figsize=(15,8))
    plt.scatter(X_train,y_train)
    plt.plot(X, lr_func(X),c="red");

    # plotting on test
    fig, ax = plt.subplots(figsize=(15,8))
    plt.scatter(X_test,y_test)
    plt.plot(X, lr_func(X),c="red");

    pre_process = PolynomialFeatures(degree=2)
    test_y_pred = lr.predict(pre_process.fit_transform(X_test))

    r2_test = r2_score(y_true=y_test, y_pred=test_y_pred )
    print(f'R2:{r2_test}')
    
    return (r2_test,lr.intercept_,lr.coef_)




