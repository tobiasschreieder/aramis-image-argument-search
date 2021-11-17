import logging

from flask import Flask, render_template, send_from_directory, request

from indexing import DataEntry
from retrieval import RetrievalSystem

log = logging.getLogger('frontend.flask')
app = Flask(__name__)

retrieval_system: RetrievalSystem = None


def start_server(system: RetrievalSystem, debug: bool = True, host: str = '0.0.0.0', port: int = 5000):
    log.debug('starting Flask')
    app.secret_key = '977e39540e424831d8731b8bf17f2484'
    app.debug = debug
    global retrieval_system
    retrieval_system = system
    app.run(host=host, port=port, use_reloader=False)


@app.route('/', methods=['POST', 'GET'])
def index():
    # if request.method == 'GET':
    #     id_list = DataEntry.get_image_ids(10)
    #     pro_images = [DataEntry.load(iid) for iid in id_list]
    #     return render_template('index.html', pros=[], cons=[])

    if request.method == 'POST':
        if retrieval_system is not None:
            if 'query' in request.form.keys():
                pro_result = retrieval_system.query(request.form['query'] + ' good', top_k=20)
                con_result = retrieval_system.query(request.form['query'] + ' anti', top_k=20)
                pro_images = [DataEntry.load(iid[0]) for iid in pro_result]
                con_images = [DataEntry.load(iid[0]) for iid in con_result]
                return render_template('index.html', pros=pro_images, cons=con_images)

    return render_template('index.html', pros=[], cons=[])


@app.route('/data/<path:name>')
def data(name):
    return send_from_directory('../data', name)
