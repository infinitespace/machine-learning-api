#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys; sys.path.insert(0, 'lib') # this line is necessary for the rest
import os                             # of the imports to work!

import web
import sqlitedb
from jinja2 import Environment, FileSystemLoader
from datetime import datetime
from web import form
import numpy as np
from sklearn.externals import joblib

# WARNING: DO NOT CHANGE THIS METHOD
def render_template(template_name, **context):
    extensions = context.pop('extensions', [])
    globals = context.pop('globals', {})
    jinja_env = Environment(autoescape=True,
            loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), 'templates')),
            extensions=extensions,)
    jinja_env.globals.update(globals)
    web.header('Content-Type','text/html; charset=utf-8', unique=True)
    return jinja_env.get_template(template_name).render(context)

# Parameters
urls = (# first parameter => URL, second parameter => class name
        '/predict', 'predict')
OUTPUT_FILE  = './models/'
MODEL_FILE  = 'svm_poly_4.pkl'

myform = form.Form( 
    form.Textbox("boe"), 
    form.Textbox("bax", 
        form.notnull,
        form.regexp('\d+', 'Must be a digit'),
        form.Validator('Must be more than 5', lambda x:int(x)>5)),
    form.Textarea('moe'),
    form.Checkbox('curly'), 
    form.Dropdown('french', ['mustard', 'fries', 'wine'])) 

render = web.template.render('templates/')

# Processing functions
def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def parse_x(X_input):
    Err = np.zeros((1, 3)) 
    try:
        tmp = X_input.split(',')
        X_len = len(tmp)
        X = np.zeros((1, X_len))     
        for j in range(X_len):
            if is_float(tmp[j]):
                X[0,j] = float(tmp[j])
            else:
                return Err
        return X
    except:
        return Err
    
# Prediction page:
class predict:
    def GET(self):
        return render_template('predict.html')
    def POST(self):
        post_params = web.input()
        X_input = post_params['X_input']
        X = parse_x(X_input)
        if len(X[0,:]) == 14:
            clf = joblib.load(OUTPUT_FILE + MODEL_FILE)
            y = clf.predict(X)
            try:
                tmp_x = str(X)
                tmp_y = str(y)
                return render_template('predict.html', parsed_x = tmp_x, predict_result = tmp_y, err_flag = False)
            except:
                err_mess = 'Render error!'
                return render_template('predict.html', err_message = err_mess, err_flag = True)
        else:
            try:
                err_mess = 'Input format error!'
                return render_template('predict.html', err_message = err_mess, err_flag = True)
            except:
                err_mess = 'Render error!'
                return render_template('predict.html', err_message = err_mess, err_flag = True)
        

# WARNING: DO NOT CHANGE THIS METHOD
if __name__ == '__main__':
    web.internalerror = web.debugerror
    app = web.application(urls, globals())
    app.add_processor(web.loadhook(sqlitedb.enforceForeignKey))
    app.run()
